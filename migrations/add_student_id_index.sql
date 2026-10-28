-- Exact Student ID lookups were reading all of active_student_data
-- (about 16,700 rows) because student_id had no index.
-- Not unique: 11 IDs are shared by two students, and the school
-- filter picks the right one.
-- The lookup compares student_id directly so this index is used.
-- Two stored IDs have a leading space; those still use the slow
-- TRIM comparison, and only when the indexed lookup misses.

CREATE INDEX idx_student_id ON active_student_data (student_id);
