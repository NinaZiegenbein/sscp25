# sscp25
Repository for the project "Cardiac MRI Statistical Shape Model Analysis" of the Simula Summer School 2025


### Virtual environment setup

You can createa a virtual environment and install all required packages using `pipenv`. If you don't already have `pipenv` installed you can install it by running the following command:

```
pip install --user pipenv
```

Make sure you have both `python` and `pip` installed before running the command. Full documentation of `pipenv` installation can be found here: [https://pipenv.pypa.io/en/latest/installation.html](URL).


To install and activate the virtual environment run the following command in your terminal where the `Pipfile` and `Pipfile.lock` are located:

```
pipenv install
```

If you just want to activate the environment without re-running installation you can run

```
pipenv shell
```

You can add new packages to the virual environment in the following way:

```
pipenv install <package-name>
```