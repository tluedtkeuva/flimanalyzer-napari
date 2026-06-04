#
# Simple script to launch_napari.py within VS Code
from napari import Viewer, run

viewer = Viewer()
# dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
#     'YOUR_PLUGIN_NAME', 'YOUR_WIDGET_NAME'
# )
# Optional steps to setup your plugin to a state of failure
# E.g. plugin_widget.parameter_name.value = "some value"
# E.g. plugin_widget.button.click()
run()
