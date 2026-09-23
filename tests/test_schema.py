import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import sequence_table

from src.data.schema import (COLUMNS, FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LENGTH, SEQUENCES,
                             TARGET_COLUMNS, WARMUP, Kind, SchemaError, detect_kind,
                             iter_sequences, sequence_ids, validate_metadata, validate_schema,
                             validate_sequence)


def test_feature_layout_matches_data_overview():
    assert len(FEATURE_COLUMNS) == N_FEATURES == len(set(FEATURE_COLUMNS))
    # Positions from the table in docs/data_overview.md.
    expected = {0: "i0_p0", 10: "i0_p10", 11: "i0_p11", 21: "i0_p21", 22: "i0_v0", 44: "i0_dp0",
                48: "i0_dv0", 52: "i1_p0", 74: "i1_v0", 96: "i1_dp0", 100: "i1_dv0", 104: "a0", 111: "a7"}
    assert {i: FEATURE_COLUMNS[i] for i in expected} == expected
    assert [len(COLUMNS[k]) for k in Kind] == [117, 118, 3]


def test_matches_starterpack_utils(official):
    assert FEATURE_COLUMNS == tuple(official.FEATURE_COLUMNS)
    assert TARGET_COLUMNS == tuple(official.TARGET_COLUMNS)
    assert (SEQUENCE_LENGTH, WARMUP, N_FEATURES) == (
        official.SEQUENCE_LENGTH, official.WARMUP, official.N_FEATURES)


def test_detect_kind_rejects_alphabetical_features():
    for kind in Kind:
        assert detect_kind(COLUMNS[kind]) is kind
    names = list(COLUMNS[Kind.TRAIN])
    names[3:3 + N_FEATURES] = sorted(FEATURE_COLUMNS)
    with pytest.raises(SchemaError, match="must not be sorted"):
        detect_kind(names)
    with pytest.raises(SchemaError, match="missing"):
        detect_kind(COLUMNS[Kind.TRAIN][:-1])
    with pytest.raises(SchemaError, match="repeated names"):
        detect_kind((*COLUMNS[Kind.TRAIN], "t1"))


def test_validate_schema_checks_types_and_kind():
    table = sequence_table(Kind.TRAIN)
    assert validate_schema(table.schema) is Kind.TRAIN
    with pytest.raises(SchemaError, match="expected a valid file"):
        validate_schema(table.schema, "valid")
    wide = table.set_column(3, "i0_p0", pa.array(np.zeros(SEQUENCE_LENGTH)))
    with pytest.raises(SchemaError, match="i0_p0: double"):
        validate_schema(wide.schema)


def test_validate_sequence_accepts_the_contract():
    assert validate_sequence(sequence_table(Kind.VALID, seq=7)) == 7
    assert validate_sequence(sequence_table(Kind.VALID_MASK, seq=3)) == 3


steps = np.arange(SEQUENCE_LENGTH)


@pytest.mark.parametrize("overrides, message", [
    ({"need_prediction": steps >= WARMUP - 1}, "warm-up"),
    ({"is_scored": steps % 2 == 0}, "warm-up rows are marked"),
    ({"seq_ix": np.where(steps < 5, 1, 7)}, "several seq_ix"),
    ({"step_in_seq": steps[::-1].copy()}, "in order"),
    ({"i1_dv3": np.where(steps == 500, np.nan, 0.0).astype(np.float32)}, "nonfinite values in i1_dv3"),
    ({"t1": np.full(SEQUENCE_LENGTH, np.inf, np.float32)}, "nonfinite values in t1"),
])
def test_validate_sequence_rejects_violations(overrides, message):
    with pytest.raises(SchemaError, match=message):
        validate_sequence(sequence_table(Kind.VALID, **overrides))
    with pytest.raises(SchemaError, match="rows"):
        validate_sequence(sequence_table(Kind.VALID).slice(0, 100))


def test_validate_metadata_proves_layout_from_footer(write_dataset, tmp_path):
    summary = validate_metadata(pq.read_metadata(write_dataset(seq_ids=(11, 5, 42))), sequences=3)
    assert summary.kind is Kind.VALID and summary.rows == 3 * SEQUENCE_LENGTH
    assert summary.seq_ix.tolist() == [11, 5, 42]
    with pytest.raises(SchemaError, match="expected 4 sequences"):
        validate_metadata(pq.read_metadata(write_dataset(seq_ids=(1, 2))), sequences=4)
    with pytest.raises(SchemaError, match=r"seq_ix \[1\] spans several row groups"):
        validate_metadata(pq.read_metadata(write_dataset(seq_ids=(1, 1), name="dup.parquet")))
    merged = tmp_path / "merged.parquet"
    pq.write_table(pa.concat_tables([sequence_table(seq=1), sequence_table(seq=2)]), merged,
                   row_group_size=2 * SEQUENCE_LENGTH)
    with pytest.raises(SchemaError, match="40000 rows"):
        validate_metadata(pq.read_metadata(merged))


def test_files_without_statistics_are_proven_by_sequence_ids(tmp_path):
    # The distributed files are written without column statistics.
    def write(name, tables):
        path = tmp_path / name
        with pq.ParquetWriter(path, tables[0].schema, write_statistics=False) as writer:
            for table in tables:
                writer.write_table(table, row_group_size=SEQUENCE_LENGTH)
        return pq.ParquetFile(path)

    good = write("good.parquet", [sequence_table(seq=4), sequence_table(seq=2)])
    summary = validate_metadata(good.metadata, Kind.VALID, sequences=2)
    assert summary.seq_ix is None
    assert sequence_ids(good).tolist() == [4, 2]
    with pytest.raises(SchemaError, match="spans several row groups"):
        sequence_ids(write("dup.parquet", [sequence_table(seq=4), sequence_table(seq=4)]))
    mixed = sequence_table(seq_ix=np.where(np.arange(SEQUENCE_LENGTH) < 10, 1, 2))
    with pytest.raises(SchemaError, match="mixes several seq_ix"):
        sequence_ids(write("mixed.parquet", [mixed]))


def test_iter_sequences_arrays(write_dataset):
    path = write_dataset(Kind.TRAIN, seq_ids=(3, 9))
    sequences = list(iter_sequences(path))
    assert [s.seq_ix for s in sequences] == [3, 9]
    first = sequences[0]
    assert first.features.shape == (SEQUENCE_LENGTH, N_FEATURES) and first.features.dtype == np.float32
    assert first.is_scored is None and first.targets.shape == (SEQUENCE_LENGTH, 2)
    table = pq.ParquetFile(path).read_row_group(0)
    assert np.array_equal(first.features[:, 104], table["a0"].to_numpy())
    subset = next(iter_sequences(path, [1], features=("a7", "i0_p0")))
    assert subset.seq_ix == 9 and subset.features.shape == (SEQUENCE_LENGTH, 2)
    with pytest.raises(SchemaError, match="not feature columns"):
        next(iter_sequences(path, features=("t0",)))


# Real starter-pack files. Footer checks cover every sequence; row checks
# cover the first few unless FULL_DATA_CHECKS=1.

def test_valid_footer_and_ids_cover_every_sequence(valid_path):
    summary = validate_metadata(pq.read_metadata(valid_path), Kind.VALID,
                                sequences=SEQUENCES[Kind.VALID])
    assert summary.rows == 37_460_000
    assert len(sequence_ids(pq.ParquetFile(valid_path))) == SEQUENCES[Kind.VALID]


def test_valid_mask_aligns_with_valid(valid_mask_path, valid_path, check_groups):
    mask, valid = pq.ParquetFile(valid_mask_path), pq.ParquetFile(valid_path)
    validate_metadata(mask.metadata, Kind.VALID_MASK, sequences=SEQUENCES[Kind.VALID_MASK])
    assert np.array_equal(sequence_ids(mask), sequence_ids(valid))
    for group in check_groups(mask.metadata.num_row_groups):
        rows = mask.read_row_group(group)
        validate_sequence(rows)
        expected = valid.read_row_group(group, columns=["is_scored"])["is_scored"]
        assert rows["is_scored"].equals(expected)


def test_train_footer_covers_every_sequence(train_metadata):
    summary = validate_metadata(train_metadata, Kind.TRAIN, sequences=SEQUENCES[Kind.TRAIN])
    assert summary.rows == 212_140_000


def test_train_data_matches_train_footer(train_path, train_metadata):
    head = validate_metadata(pq.read_metadata(train_path), Kind.TRAIN)
    full = validate_metadata(train_metadata, Kind.TRAIN)
    assert 0 < head.sequences <= full.sequences
    assert pq.read_schema(train_path).equals(train_metadata.schema.to_arrow_schema(),
                                             check_metadata=False)
    assert len(sequence_ids(pq.ParquetFile(train_path))) == head.sequences


def test_real_rows_satisfy_contract(valid_path, train_path, check_groups):
    for path in (valid_path, train_path):
        groups = check_groups(pq.ParquetFile(path).metadata.num_row_groups)
        for seq in iter_sequences(path, groups):
            assert int(seq.need_prediction.sum()) == SEQUENCE_LENGTH - WARMUP
            if seq.is_scored is not None:
                assert seq.is_scored.any() and not (seq.is_scored & ~seq.need_prediction).any()
