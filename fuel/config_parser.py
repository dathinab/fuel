"""Module level configuration.

Fuel allows module-wide configuration values to be set using a YAML_
configuration file and `environment variables`_. Environment variables
override the configuration file which in its turn overrides the defaults.

The configuration is read from ``~/.fuelrc`` if it exists. A custom
configuration file can be used by setting the ``FUEL_CONFIG`` environment
variable. A configuration file is of the form:

.. code-block:: yaml

   data_path: /home/user/datasets

Which could be overwritten by using environment variables:

.. code-block:: bash

   $ FUEL_DATA_PATH=/home/users/other_datasets python

This data path is a sequence of paths separated by an os-specific
delimiter (':' for Linux and OSX, ';' for Windows).

If a setting is not configured and does not provide a default, a
:class:`~.ConfigurationError` is raised when it is
accessed.

Configuration values can be accessed as attributes of
:const:`fuel.config`.

    >>> from fuel import config
    >>> print(config.data_path) # doctest: +SKIP
    '~/datasets'

The following configurations are supported:

.. option:: data_path

   The path where dataset files are stored. Can also be set using the
   environment variable ``FUEL_DATA_PATH``. Expected to be a sequence
   of paths separated by an os-specific delimiter (':' for Linux and
   OSX, ';' for Windows).

.. option:: default_seed

    the default rng seed used by some randomness including schemes, converters etc.

.. option:: floatX

   The default :class:`~numpy.dtype` to use for floating point numbers. The
   default value is ``float64``. A lower value can save memory, and speed
   up/enable computations on a GPU. Note that if ``fuel_uses_theano`` is set
   it will import theano and use ``theano.config.floatX`` as default value insted
   of ``float64``

.. option:: fuel_uses_theano

    Specifies if fuel is allowed to/should use theano. Defaults to True. If this
    value is set theano will be imported and it will use the theanos config for
    finding a default for floatX. Other use cases might be possible, too.

.. option:: extra_downloaders

   A list of package names which, like fuel.downloaders, define an
   `all_downloaders` attribute listing available downloaders. By default,
   an empty list.

.. option:: extra_converters

   A list of package names which, like fuel.converters, define an
   `all_converters` attribute listing available converters. By default,
   an empty list.

.. _YAML: http://yaml.org/
.. _environment variables:
   https://en.wikipedia.org/wiki/Environment_variable

"""
import logging
import os
import sys

import six
import yaml
import functools

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

NOT_SET = object()


def extra_downloader_converter(value):
    """Parses extra_{downloader,converter} arguments.

    Parameters
    ----------
    value : iterable or str
        If the value is a string, it is split into a list using spaces
        as delimiters. Otherwise, it is returned as is.

    """
    if isinstance(value, six.string_types):
        value = value.split(" ")
    return value


def multiple_paths_parser(value):
    """Parses data_path argument.

    Parameters
    ----------
    value : str
        a string of data paths separated by  ":".

    Returns
    -------
    value : list
        a list of strings indicating each data paths.

    """
    if isinstance(value, six.string_types):
        value = value.split(os.path.pathsep)
    return value


class Configuration(object):
    def __init__(self):
        self.config = {}

    def load_yaml(self):
        if 'FUEL_CONFIG' in os.environ:
            yaml_file = os.environ['FUEL_CONFIG']
        else:
            yaml_file = os.path.expanduser('~/.fuelrc')
        if os.path.isfile(yaml_file):
            with open(yaml_file) as f:
                for key, value in yaml.safe_load(f).items():
                    if key not in self.config:
                        raise ValueError("Unrecognized config in YAML: {}"
                                         .format(key))
                    self.config[key]['yaml'] = value

    def __getattr__(self, key):
        if key == 'config' or key not in self.config:
            raise AttributeError
        config_setting = self.config[key]
        if 'value' in config_setting:
            value = config_setting['value']
        elif ('env_var' in config_setting and
              config_setting['env_var'] in os.environ):
            value = os.environ[config_setting['env_var']]
        elif 'yaml' in config_setting:
            value = config_setting['yaml']
        elif 'default' in config_setting:
            value = config_setting['default']
        else:
            raise ConfigurationError("Configuration not set and no default "
                                     "provided: {}.".format(key))
        return config_setting['type'](value)

    def __setattr__(self, key, value):
        if key != 'config' and key in self.config:
            self.config[key]['value'] = value
        else:
            super(Configuration, self).__setattr__(key, value)

    def add_config(self, key, type_, default=NOT_SET, env_var=None):
        """Add a configuration setting.

        Parameters
        ----------
        key : str
            The name of the configuration setting. This must be a valid
            Python attribute name i.e. alphanumeric with underscores.
        type : function
            A function such as ``float``, ``int`` or ``str`` which takes
            the configuration value and returns an object of the correct
            type.  Note that the values retrieved from environment
            variables are always strings, while those retrieved from the
            YAML file might already be parsed. Hence, the function provided
            here must accept both types of input.
        default : object, optional
            The default configuration to return if not set. By default none
            is set and an error is raised instead.
        env_var : str, optional
            The environment variable name that holds this configuration
            value. If not given, this configuration can only be set in the
            YAML configuration file.

        """
        self.config[key] = {'type': type_}
        if env_var is not None:
            self.config[key]['env_var'] = env_var
        if default is not NOT_SET:
            self.config[key]['default'] = default


def _human_bool(value, default):
    """casts to bool, interpreting strings semantically instead of length based

    String interpretations
    ----------------------
    - true    is 'True', '1' or 'true'
    - false   is 'False', '0' or 'false'
    - default is ''

    Other types
    -----------
    return bool(value)

    :param value: the value to interprete as bool
    :param default: the default value used for empty strings
    :return: a boolean interpretation of the value
    :raises ValueError: if the value is a string and '' or a boolean description like 'True'/'False'
    """

    if isinstance(value, str):
        if len(value) == 0:
            return default
        if value == "1" or value == "True" or value == "true":
            return True
        elif value == "0" or value == "False" or value == "false":
            return False
        else:
            raise ValueError("string will not be interpreted as bool use 'True'/'False' instead")
    return bool(value)


config = Configuration()

# Define configuration options
config.add_config('data_path', type_=multiple_paths_parser,
                  env_var='FUEL_DATA_PATH')
config.add_config('default_seed', type_=int, default=1)
config.add_config('extra_downloaders', type_=extra_downloader_converter,
                  default=[], env_var='FUEL_EXTRA_DOWNLOADERS')
config.add_config('extra_converters', type_=extra_downloader_converter,
                  default=[], env_var='FUEL_EXTRA_CONVERTERS')

# the default of floatX might be overriden if `fuel_uses_theano` is Tru(ly) AFTER user configs are loaded
config.add_config('floatX', type_=str, env_var='FUEL_FLOATX', default='float64')
# not that you have to set fuel_uses_theano=False if theano might not be available
config.add_config('fuel_uses_theano', type_=functools.partial(_human_bool, default=True), env_var='FUEL_USES_THEANO',
                  default=True)

config.load_yaml()

# under some cicumstances wen want to use the thano floatX config as default
# but we NEVER want to load theano unconditionally as it will bind/initialize the GPU, CUDA etc.
if config.fuel_uses_theano:
    from theano import config as theano_config
    # override the default, not the user configuration
    config.config['floatX']['default'] = theano_config.floatX
