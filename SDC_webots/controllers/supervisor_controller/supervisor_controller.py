"""supervisor_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor

TIME_STEP = 32

# create the Robot instance.
robot = Supervisor()

# get vehicle's node (DEF VEHICLE en el .wbt)
vehicle_node = robot.getFromDef('VEHICLE')

# NOTA (Actividad 4.1): se removieron los peatones y el barril del mundo para
# dejar solo el escenario y el coche. El supervisor ya NO lanza barriles
# (antes hacia getFromDef('BARREL') y lo reposicionaba cerca del coche);
# ahora unicamente avanza la simulacion.

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    pass

# Enter here exit cleanup code.
