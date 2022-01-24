# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.

from numpy import isclose

from matplotlib.ticker import NullLocator
from matplotlib.ticker import NullFormatter, ScalarFormatter, LogFormatterSciNotation

from mantid.plots.utility import convert_color_to_hex, get_autoscale_limits
from workbench.plotting.plotscriptgenerator.utils import convert_value_to_arg_string

BASE_AXIS_LABEL_COMMAND = "set_{}label({})"
BASE_AXIS_LIM_COMMAND = "set_{}lim({})"
BASE_SET_TITLE_COMMAND = "set_title({})"
BASE_AXIS_SCALE_COMMAND = "set_{}scale('{}')"
BASE_SET_FACECOLOR_COMMAND = "set_facecolor('{}')"

TICK_FORMATTER_CLASSES = {
    "NullFormatter": NullFormatter,
    "ScalarFormatter": ScalarFormatter,
    "LogFormatterSciNotation": LogFormatterSciNotation
}

TICK_FORMATTERS = {
    "NullFormatter": "NullFormatter()",
    "ScalarFormatter": "ScalarFormatter(useOffset=True)",
    "LogFormatterSciNotation": "LogFormatterSciNotation()"
}

DEFAULT_TICK_FORMATTERS = {
    "linear": {"major": ScalarFormatter, "minor": NullFormatter},
    "log": {"major": LogFormatterSciNotation, "minor": LogFormatterSciNotation}
}


def generate_axis_limit_commands(ax):
    """Generate commands to set the axes' limits"""
    commands = []
    for axis in ['x', 'y']:
        current_lims = getattr(ax, "get_{}lim".format(axis))()
        default_lims = get_autoscale_limits(ax, axis)
        if not isclose(current_lims, default_lims, rtol=0.01).all():
            arg_string = convert_value_to_arg_string(current_lims)
            commands.append(BASE_AXIS_LIM_COMMAND.format(axis, arg_string))
    return commands


def generate_axis_label_commands(ax):
    commands = []
    for axis in ['x', 'y']:
        label = getattr(ax, 'get_{}label'.format(axis))()
        if label:
            commands.append(BASE_AXIS_LABEL_COMMAND.format(axis, repr(label)))
    return commands


def generate_set_title_command(ax):
    return BASE_SET_TITLE_COMMAND.format(repr(ax.get_title()))


def generate_axis_scale_commands(ax):
    commands = []
    for axis in ['x', 'y']:
        scale = getattr(ax, 'get_{}scale'.format(axis))()
        if scale != 'linear':
            commands.append(BASE_AXIS_SCALE_COMMAND.format(axis, scale))
    return commands


def generate_axis_facecolor_commands(ax):
    return BASE_SET_FACECOLOR_COMMAND.format(convert_color_to_hex(ax.get_facecolor()))


def generate_tick_params_kwargs(axis, tick_type="major"):
    return getattr(axis, f"_{tick_type}_tick_kw")


def generate_tick_formatter_commands(ax):
    """
    Generate commands for setting tick label formats.
    """
    commands = []
    for axis in ["xaxis", "yaxis"]:
        for tick_type in ["major", "minor"]:  # Currently there is no way to change minor tick format in GUI
            formatter = getattr(getattr(ax, axis), f"get_{tick_type}_formatter")()
            # Don't write the command to the script if it's default.
            if _is_default_tick_formatter(ax, axis, tick_type, formatter):
                continue
            for key, value in TICK_FORMATTERS.items():
                if isinstance(formatter, TICK_FORMATTER_CLASSES[key]):
                    commands.append(f"{axis}.set_{tick_type}_formatter({value})")
    return commands


def generate_tick_commands(ax):
    commands = []
    for tick_type in ["minor", "major"]:
        if not isinstance(getattr(ax.xaxis, tick_type).locator, NullLocator):
            if tick_type == "minor":
                commands.append("minorticks_on()")

            if isinstance(getattr(ax.xaxis, f"{tick_type}Ticks"), list) and \
                    len(getattr(ax.xaxis, f"{tick_type}Ticks")) > 0:
                commands.append(f"tick_params(axis='x', which='{tick_type}', **"
                                f"{generate_tick_params_kwargs(ax.xaxis, tick_type)})")
                commands.append(f"tick_params(axis='y', which='{tick_type}', **"
                                f"{generate_tick_params_kwargs(ax.yaxis, tick_type)})")

    return commands


def _is_default_tick_formatter(ax, axis, tick_type, formatter):
    """
    Check whether the formatter is default for the given axis and tick type.
    """
    scale = getattr(getattr(ax, axis), "get_scale")()
    if scale in DEFAULT_TICK_FORMATTERS:
        if isinstance(formatter, DEFAULT_TICK_FORMATTERS[scale][tick_type]):
            return True
    return False
