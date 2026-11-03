RELEASE_TYPE: minor

This release adds :class:`~hypothesis.database.CompressedExampleDatabase`, a
wrapper which transparently compresses large values saved to another example
database with zlib, reducing the storage used by large failing examples. Only
values of at least a configurable minimum size are compressed, and values which
do not compress well are stored unchanged. Fetching, deleting, moving, and
change listening always operate on the original, uncompressed bytes, while
keys and data written directly to the wrapped database are left untouched.
