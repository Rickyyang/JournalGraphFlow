import math
import pygame as pg
import numpy as np
from pygame.math import Vector2
import time
import quadprog
import cvxopt
from cvxopt import matrix
import matplotlib.pyplot as plt

#r0 is the radius of the task region
#(x0, y0) is the position of the center of the task region
r0=50
x0=250
y0=250

#t_max is the deadline
#v_max is the maximum velocity
t_max=15
v_max=15

#w1, w2 are weights, see equation (14) in the paper
w1=0.5
w2=10

# beta is a weight, see equation (12)
beta=0.03

#lamda is a weight for adjusting the angle speed
lamda=0.3

def theta_fi(theta,fi):
    if abs(theta-fi)<=180:
        result=theta-fi
    elif theta>fi:
        result=theta-fi-360
    else:
        result=theta-fi+360
    return result

#convex optimization solver
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return numpy.array(sol['x']).reshape((P.shape[1],))

#quadratic program solver
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def distance(x):
    return ((x[0]-x0)**2+(x[1]-y0)**2)**0.5-r0

class Player(pg.sprite.Sprite):
    # initialize the robot as a circle
    def __init__(self, pos=(420, 420), rgb=(50,120,180),width=0):
        super(Player, self).__init__()
        self.image = pg.Surface((30, 30), pg.SRCALPHA)
        pg.draw.circle(self.image, rgb, (15,15),15,width)
        pg.draw.circle(self.image,(150,30,30),(26,15),4,width)
        self.original_image = self.image
        self.rect = self.image.get_rect(center=pos)
        self.position = Vector2(pos)
        self.direction = Vector2(1, 0)  # A unit vector pointing rightward.
        self.speed = 0
        self.angle_speed = 0
        self.angle = 0
        self.alpha=0
        self.weight=0
        self.alphadot=0

    # see equation (7) in the paper
    def h_c(self,t):
        return t_max-t-distance(self.position)/v_max  

    # update the pose of the robot
    def update(self):
        if self.angle_speed != 0:
            # Rotate the direction vector and then the image.
            self.direction.rotate_ip(self.angle_speed)
            self.angle = np.mod(self.angle+self.angle_speed,360)
            self.image = pg.transform.rotate(self.original_image, -self.angle)
            self.rect = self.image.get_rect(center=self.rect.center)
        # Update the position vector and the rect.
        self.position += self.direction * self.speed
        self.rect.center = self.position

    def fi(self,dy,dx):
        fi=np.arctan2(dy,dx)
        fi=np.rad2deg(fi)
        if fi<0:
            fi+=360
        return fi
   
def main(initial_pose):
    pg.init()

    screen = pg.display.set_mode((500, 500))
    screen.fill((255,255,255))
    
    length=len(initial_pose)  
  
    player = []
    playersprite=[]
    out_player = []
    out_playersprite = []

    for idx in range(len(initial_pose)):  
        player.append(Player((initial_pose[idx][0],initial_pose[idx][1])))
        playersprite.append(pg.sprite.RenderPlain((player[idx])))

    sum_w=0
    delete_idx=[]
    for i in range(len(player)): 
        if player[i].h_c(0)<=0:
            delete_idx.append(i)
            out_player.append(player[i])
            out_playersprite.append(playersprite[i])
        else:
            player[i].weight=player[i].h_c(0)
            sum_w = sum_w + player[i].h_c(0)
    for index in sorted(delete_idx, reverse=True):
        del player[index]
        del playersprite[index]
        
    for i in range(len(player)): 
        player[i].alpha=player[i].h_c(0)/sum_w
    clock = pg.time.Clock()
    done = False
    t0=time.time()
    
    points=[[] for _ in range(length)]
    a=[[] for _ in range(length)]
    out_a=[]
    t=[]
    out_points = []
    Closed=False
    
    while not done:
        
        clock.tick(60)  
        t1=time.time()-t0
        weight_sum=0
        t.append(t1)
        for i in range(len(player)):  
            points[i].append((player[i].position[0],player[i].position[1]))
            
            a[i].append(player[i].alpha)
            
            # calculate P,G,q,h for quadratic program
            P=w1*np.identity(2)
            G=-np.array([(player[i].position[0]-x0)/(distance(player[i].position)+r0), (player[i].position[1]-y0)/(distance(player[i].position)+r0)]).reshape(1,2)
            q=-1*player[i].alpha*w2*G.reshape(2,)/v_max
            h=np.array([(-1+player[i].h_c(t1))*v_max])
            u=quadprog_solve_qp(P, q, G, h, A=None, b=None)
            if i==0:
                print(np.dot(G,np.array(u).reshape(2,1))-h)
            player[i].weight= player[i].alpha*player[i].h_c(t1)    
            weight_sum = weight_sum + player[i].weight  
            #cvxopt_solve_qp(P, q, G, h, A=None, b=None)
            
            delete_idx=[]
            if u[0]**2+u[1]**2 >= 2:
                '''
                player[i].speed=0
                player[i].angle_speed=0
                out_player.append(player[i])
                out_playersprite.append(playersprite[i])
                out_points.append(points[i])
                out_a.append(a[i])
                delete_idx.append(i)
                '''
            else:
                player[i].angle_speed=-1*lamda*(theta_fi(player[i].angle, player[i].fi(u[1],u[0]))) 
                player[i].speed=(u[0]**2+u[1]**2)**0.5
           
        for i in range(len(player)):  
            playersprite[i].update()
            player[i].alphadot=(len(player)*player[i].weight-weight_sum)*beta
            
        screen.fill((255,255,255))  
        for i in range(len(out_player)):
            out_playersprite[i].draw(screen)   
        for i in range(len(player)):
            playersprite[i].draw(screen)         
        
        radius=int(v_max*(t_max-time.time()+t0))
        pg.draw.circle(screen, (0,0,0), (x0,y0), radius,1)
        pg.draw.circle(screen, (0,0,0), (x0,y0), r0,1)
        pg.display.flip()
         
        remain=0
        delta_t=time.time()-t1-t0
        
        ''' if the splitting coefficient is small enough or the robot isn't in the safety set
                stop the robot, add its splitting coefficient to the variable "remain" and set the splitting coefficient to 0
                split the value of "remain" to the remaining robots averagely to make sure the sum of the splitting coefficient
                equal to 1
        '''
        for i in range(len(player)):
            test=player[i].alpha+player[i].alphadot*delta_t
            if test <= 0.05 or player[i].h_c(time.time()-t0)<=0:
                player[i].speed=0
                player[i].angle_speed=0
                remain=remain+player[i].alpha
                player[i].alpha=0
                out_player.append(player[i])
                out_playersprite.append(playersprite[i])
                out_points.append(points[i])
                out_a.append(a[i])
                delete_idx.append(i) 
            else :
                player[i].alpha=test 
        
        delete_idx = list(dict.fromkeys(delete_idx))
        for index in sorted(delete_idx, reverse=True):   
            del player[index]
            del playersprite[index]
            del points[index]
            del a[index]
           
        for i in range(len(player)):
            player[i].alpha+=remain/len(player)
            if distance(player[i].position) < -15:
               player[i].speed=0
               player[i].angle_speed=0
               done=True
               pg.draw.aalines(screen, (20,200,20), Closed, points[i])
               del points[i]
               break   
        
    for i in out_points:     
        if len(i)<=1:
            continue
        pg.draw.aalines(screen, (200,20,20), Closed, i)
    
    for i in points:
        if len(i)==0:
            continue
        pg.draw.aalines(screen, (200,20,20), Closed, i)
       
    pg.display.flip()
    pg.image.save(screen, './image_1.png')
    
    index=0
    SUM=[0]*len(t)
    for i in range(len(out_a)):
        out_a[i].extend([0]*(len(t)-len(out_a[i])))
        plt.plot(t,out_a[i],label='robot_'+str(index))
        SUM=np.add(SUM,out_a[i]).tolist()
        index+=1
    for i in range(len(a)):
        a[i].extend([0]*(len(t)-len(a[i])))
        plt.plot(t,a[i],label='robot_'+str(index))
        SUM=np.add(SUM,a[i]).tolist()
        index+=1
        
    plt.plot(t,SUM,label='sum')
    plt.xlabel('t - axis') 
    plt.ylabel('\u03B1 - axis') 
    plt.title('\u03B1 - t graph') 
    plt.legend(loc="best") 
    plt.show() 
    
 
if __name__ == '__main__':
    
    #set the initial pose for the robots
    initial_pose=[[30,350],[180,350],[120,250],[400,300],[400,180]]
    main(initial_pose)
    while (True):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
    
