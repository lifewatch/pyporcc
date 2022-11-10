import setuptools

setuptools.setup(name='pyporcc',
                 version='0.1.3',
                 description='Adapted PorCC to python',
                 url='https://github.com/cparcerisas/pyporcc.git',
                 author='Clea Parcerisas',
                 author_email='cleaparcerisas@gmail.com',
                 license='',
                 package_data={
                    "pyporcc": ["data/standard_click.wav", "models/*.ini"],
                 },
                 packages=setuptools.find_packages(),
                 zip_safe=False)
