################################################################################
#                                                                              #
#  Date created: 2022/02/25                                                    #
#  Last modified: 2022/02/25                                                   #
#                                                                              #
################################################################################

import math
import matplotlib
matplotlib.use('agg',force=True)
from matplotlib import animation as anim
from matplotlib import pyplot as plt
import numpy as np
import os
import random

def run():
    
    # Initialization
    mapsize = 100
    colony_count = 20
    endtime = 1000
    tsteps = 0
    colonies = []
    distances = np.zeros((colony_count, colony_count))
    ims = []
    try:
        os.mkdir('yoink')
    except OSError:
        pass


    for n in range(colony_count):
        newpos = (10*random.random() + 45, 10*random.random() + 45)
        colonies.append(create_colony(newpos, n))
    
    for colony1 in colonies:
        for colony2 in colonies:
            x = colony1.x - colony2.x
            y = colony1.y - colony2.y
            distances[colony1.id, colony2.id] = math.sqrt(x ** 2 + y ** 2)

    while tsteps < endtime:
        tsteps += 1
        # Plot setup
        for colony in colonies:

            migration = colony.migration_check(distances)
            if migration:
                for colony1 in colonies:
                    x = colony.x - colony1.x
                    y = colony.y - colony1.y
                    newdistance = math.sqrt(x ** 2 + y ** 2)
                    distances[colony.id, colony1.id] = newdistance
                    distances[colony1.id, colony.id] = newdistance

            spawn = colony.spawn_check()
            if spawn:
                maxradius = 3
                stepsize = maxradius * random.random()
                direction = 2 * math.pi * random.random()
                x = colony.x + stepsize * math.cos(direction)
                y = colony.y + stepsize * math.sin(direction)
                n = len(colonies)
                colonies.append(create_colony((x,y), n))
                distances = np.insert(distances, -1, 0, axis=1)
                distances = np.append(distances, [np.zeros(len(colonies))],
                                        axis=0)
                colony = colonies[n]
                for colony1 in colonies:
                    newdistance = math.sqrt(x ** 2 + y ** 2)
                    distances[colony.id, colony1.id] = newdistance
                    distances[colony1.id, colony.id] = newdistance
        
        m = 0
        while m < len(colonies):
            death = colonies[m].death_check()
            if death:
                colonies.remove(colonies[m])
                distances = np.delete(distances, m, axis=0)
                distances = np.delete(distances, m, axis=1)
                for colony in colonies:
                    if colony.id > m:
                        colony.id -= 1
                    else:
                        pass
            else:
                # Draw
                plot = plt.plot([colonies[m].x], [colonies[m].y], marker=".")
                plt.xlim(0,mapsize)
                plt.ylim(0,mapsize)
                m += 1

        ims.append(plot)
        #fig.savefig('yoink/frame{}.png'.format(tsteps))
        plt.close('all')
    fig = plt.figure()
    ax = fig.subplots()
    ani = anim.ArtistAnimation(fig, ims)
    ani.save('yoink/montage.mp4', fps = 15)
class create_colony:
    
    def __init__(self, pos, idnumber):
        self.x, self.y = pos
        self.migration_rate = 0.06
        self.id = idnumber
        self.repulsion_coef = 0.5

    def migration_check(self, distances):
        migrate_chance = 0
        for n in range(len(distances)):
            if n != self.id:
                distance = distances[n,self.id]
                repulsion = self.repulsion_coef / (distance+0.000001) ** 2
                migrate_chance += self.migration_rate * repulsion
        if random.random() < migrate_chance:
            self.migrate()
            return True
        else:
            return False

    def migrate(self):
        maxradius = 3
        stepsize = maxradius * random.random()
        direction = 2 * math.pi * random.random()
        self.x += stepsize * math.cos(direction)
        self.y += stepsize * math.sin(direction)
    
    def spawn_check(self):
        spawn_chance = 0.002
        if random.random() < spawn_chance:
            return True
        else:
            return False

    def spawn(self):
        maxradius = 4
        stepsize = maxradius * random.random()
        direction = 2 * math.pi * random.random()
        self.x += stepsize * math.cos(direction)
        self.y += stepsize * math.sin(direction)

    def death_check(self):
        death_chance = 0.001
        if random.random() < death_chance:
            return True
        else:
            return False

if __name__ == '__main__':
    run()
