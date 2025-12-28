import keyboard
import numpy as np
import pygame as pg

class Teleop2D:
    def __init__(self):
        self.dir_bindings = {
            'i': (1,0,0), # forward
            ',': (-1,0,0), # backward
            'J': (0,1,0), # strafe left
            'L': (0,-1,0), # strafe right
            'k': (0,0,0), # stop
            'j': (0,0,1), # ccw
            'l': (0,0,-1), # cw
        }
        self.speed_bindings = {
            'w': (1.1,1.0), # inc
            's': (0.9,1.0), # dec
            'a': (1.0,1.1), # inc
            'd': (1.0,0.9) # dec
        }
        self.speed = 0.5
        self.yaw_speed = 1.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self._print_instructions()

    def _print_instructions(self):
        msg = """
        Teleop2D Keyboard Control
        -------------------------
        Movement:
          i : forward
          , : backward
          J : strafe left
          L : strafe right
          j : rotate ccw
          l : rotate cw
          k : stop

        Speed control:
          w : increase linear speed
          s : decrease linear speed
          a : increase yaw speed
          d : decrease yaw speed

        Current speed: {lin:.2f}, Yaw speed: {yaw:.2f}
        -------------------------
        """.format(lin=self.speed, yaw=self.yaw_speed) 
        print(msg)
    
    def advance(self):
        u = 0
        v = 0
        r = 0
        key = keyboard.read_key()
        if key in self.dir_bindings.keys():
            u = self.dir_bindings[key][0]
            v = self.dir_bindings[key][1]
            r = self.dir_bindings[key][2]
        elif key in self.speed_bindings.keys():
            self.speed = self.speed_bindings[key][0]*self.speed
            self.yaw_speed = self.speed_bindings[key][1]*self.yaw_speed
        self.u = u*self.speed
        self.v = v*self.speed
        self.r = r*self.yaw_speed
        return np.array([self.u, self.v, self.r])
    

class Keyboard2D:
    def __init__(self):
        self.dir_bindings = {
            'i': (1,0,0), # forward
            ',': (-1,0,0), # backward
            'J': (0,1,0), # strafe left
            'L': (0,-1,0), # strafe right
            'k': (0,0,0), # stop
            'j': (0,0,1), # ccw
            'l': (0,0,-1), # cw
        }

        self.speed_bindings = {
            'w': (1.1,1.0), # inc
            's': (0.9,1.0), # dec
            'a': (1.0,1.1), # inc
            'd': (1.0,0.9) # dec
        }
        self.speed = 0.5
        self.yaw_speed = 1.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self._print_instructions()

    def _print_instructions(self):
        msg = """
        Teleop2D Keyboard Control
        -------------------------
        Movement:
          i : forward
          , : backward
          J : strafe left
          L : strafe right
          j : rotate ccw
          l : rotate cw
          k : stop

        Speed control:
          w : increase linear speed
          s : decrease linear speed
          a : increase yaw speed
          d : decrease yaw speed

        Current speed: {lin:.2f}, Yaw speed: {yaw:.2f}
        -------------------------
        """.format(lin=self.speed, yaw=self.yaw_speed)
        print(msg)

    def advance(self):
        u = 0
        v = 0
        r = 0
        if pg.display.get_surface() is None:
            return
        
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                key = pg.key.name(event.key)
                if event.key == pg.K_l or event.key == pg.K_j:
                    if event.mod & pg.KMOD_SHIFT:
                        key = key.upper()
                    else:
                        key = key
                if key in self.dir_bindings.keys():
                    u = self.dir_bindings[key][0]
                    v = self.dir_bindings[key][1]
                    r = self.dir_bindings[key][2]
                elif key in self.speed_bindings.keys():
                    self.speed *= self.speed_bindings[key][0]
                    self.yaw_speed *= self.speed_bindings[key][1]

                self.u = u*self.speed
                self.v = v*self.speed
                self.r = r*self.yaw_speed

        return np.array([self.u, self.v, self.r])




