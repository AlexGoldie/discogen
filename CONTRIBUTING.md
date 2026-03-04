# Contributing to `DiscoGen`

The utility of DiscoGen dramatically increases as we improve the number and diversity of tasks available within the repository. As such, there are three types of contributions to DiscoGen:

## New Domains

If you have an interesting idea for a new task that can fit the DiscoGen style, please add it (following the guidance in [our documentation](/docs/how_to/overview.md)). We will endeavour to add all well-implemented tasks to make sure DiscoGen remains a live and fresh benchmark with significant open-source contribution.

## Improvements To Domains

Have you noticed more interesting modules in a pre-existing task than the original author? Great! Please feel free to separate out that module from the original codebase into a `base` and `edit` implementation! Simply increasing the number of modules in a codebase from 3 -> 4 increases the number of module combinations from 5 -> 23!

## Bug-Fixes

If you spot any bugs in DiscoGen, please raise an issue and we will see to fix it as soon as possible!


# New to using Ruff and Mypy?

If you have not used ruff and mypy in your VS Code before, you can follow the following steps to set up a good development environment.

1. Install `charliermarsh/Ruff` and `ms-python/Mypy Type Checker` extension.
2. Open you workspace settings and paste the following:
```
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "mypy-type-checker.importStrategy": "fromEnvironment",
    "mypy-type-checker.reportingScope": "workspace",
    "mypy-type-checker.preferDaemon": true,
    "ruff.nativeServer": "on",
    "ruff.enable": true,
    "ruff.lint.enable": true,
    "ruff.organizeImports": true,
    "ruff.importStrategy": "fromEnvironment",
    "ruff.fixAll": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.formatOnSaveMode": "file",
        "editor.codeActionsOnSave": {
            "source.fixAll": "explicit",
            "source.organizeImports": "explicit"
        },
    },
    "cursorpyright.analysis.typeCheckingMode": "off",
    "cursorpyright.analysis.exclude": ["**"],
    "makefile.configureOnOpen": false,
    "python.languageServer": "None",
}
```

These settings are meant to disable the annoying and redundant pyright/pylance errors and enable some sane defaults.

# Get Started!

Ready to contribute? Here's how to set up `discogen` for local development.
Please note this documentation assumes you already have `uv` and `Git` installed and ready to go.

1. Fork the `discogen` repo on GitHub.

2. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/discogen.git
```

3. Now we need to install the environment. Navigate into the directory

```bash
cd discogen
```

Then, install and activate the environment with:

```bash
uv sync --all-extras
```

4. Install pre-commit to run linters/formatters at commit time:

```bash
uv run pre-commit install
```

5. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

6. Don't forget to add test cases for your added functionality to the `tests` directory.

7. When you're done making changes, check that your changes pass the formatting tests.

```bash
make check
```

Now, validate that all unit tests are passing:

```bash
make test
```

9. Before raising a pull request you should also run tox. (Optional)
   This will run the tests across different versions of Python:

```bash
tox
```

This requires you to have multiple versions of python installed.
This step is also triggered in the CI/CD pipeline, so you could also choose to skip this step locally.

10. Commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

11. Submit a pull request through the GitHub website.

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
