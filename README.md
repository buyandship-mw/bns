**Running Scripts**

To run any script in this repository, use the following pattern from the repo root (`bns/`):

```sh
python3 -m bns.<project>.<script>
```

For example, to run `app.py` in `proj2`:

```sh
python3 -m bns.proj2.app
```

> **Note:** Replace `<project>` and `<script>` with the appropriate module and script name you wish to run.

This approach ensures that all package imports work as expected.