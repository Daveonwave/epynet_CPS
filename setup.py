import setuptools

setuptools.setup(
    name="epynetCPS",
    version='0.0.1',
    packages=['epynet_cps.main'],
    install_requires=[
        'wntr',
        'pandas',
        'schema',
        'tqdm'
    ],
    python_requires=">=3.8",
    entrypoints={
        'console_scripts': [
            'epynet = epynet_cps.main.command_line:main'
        ]
    }
)
