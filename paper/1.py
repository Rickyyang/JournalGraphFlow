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

# 1,2 represents the obstalces
r1=25
x1=250
y1=320

r2=30
x2=180
y2=325

r4=40
x4=320
y4=350

r5=40
x5=120
y5=200

#3 represent the connectivity circle
r3=400
x3=250
y3=350

#t_max is the deadline
#v_max is the maximum velocity
t_max=20
v_max=15

#w1, w2 are weights, see equation (14) in the paper
w1=0.1
w2=0.5
w3=0.01

# beta is a weight, see equation (12)
beta=0.06

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
    def __init__(self, pos=(420, 420), rgb=(50,120,180),width=0, radius=15, obs=False):
        super(Player, self).__init__()
        self.image = pg.Surface((radius*2, radius*2), pg.SRCALPHA)
        pg.draw.circle(self.image, rgb, (radius,radius),radius,width)
        if obs==False:
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

    def h_d(self, pos, r_d):
        return (self.position[0]- pos[0])**2+(self.position[1]- pos[1])**2-(r_d+15)**2

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
   
def main(initial_pose, rgb):
    pg.init()

    screen = pg.display.set_mode((500, 500))
    screen.fill((255,255,255))
    
    length=len(initial_pose)  
  
    player = []
    playersprite=[]
    out_player = []
    out_playersprite = []

    for idx in range(len(initial_pose)):  
        player.append(Player((initial_pose[idx][0],initial_pose[idx][1]),rgb=rgb[idx],width=4))
        playersprite.append(pg.sprite.RenderPlain((player[idx])))

    obs_1=Player((x1,y1),rgb=(200,200,200),radius=r1, obs=True)
    obs_playersprite_1=pg.sprite.RenderPlain((obs_1))
    obs_1.angle=60

    obs_2=Player((x2,y2),rgb=(200,200,200),radius=r2, obs=True)
    obs_playersprite_2=pg.sprite.RenderPlain((obs_2))
    obs_2.angle=60

    obs_3=Player((x5,y5),rgb=(200,200,200),radius=r5, obs=True)
    obs_playersprite_3=pg.sprite.RenderPlain((obs_3))
    obs_3.angle=60

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
    counter=0
    current_pose=[None]*length
    last_pose=[]
    t1=time.time()-t0
    for i in range(length):
          player[i].weight= player[i].alpha*player[i].h_c(t1)
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
            G_1=2*np.array([player[i].position[0]-x3, player[i].position[1]-y3]).reshape(1,2)
            G_2=-np.array([(player[i].position[0]-x0)/(distance(player[i].position)+r0), (player[i].position[1]-y0)/(distance(player[i].position)+r0)]).reshape(1,2)
            G_3=2*np.array([x1-player[i].position[0], y1-player[i].position[1]]).reshape(1,2)
            G_4=2*np.array([x2-player[i].position[0], y2-player[i].position[1]]).reshape(1,2)
            G_5=2*np.array([x5-player[i].position[0], y5-player[i].position[1]]).reshape(1,2)
            h_1=r3**2-(player[i].position[0]-x3)**2-(player[i].position[1]-y3)**2
            h_2=np.array([(-1+player[i].h_c(t1))*v_max])
            h_3=player[i].h_d(obs_1.position, r1)
            h_4=player[i].h_d(obs_2.position, r2)
            h_5=player[i].h_d(obs_3.position, r5)
            G=np.vstack((G_1,G_2,G_3,G_4,G_5))
            #G_D_1=-np.array([player[i].position[0]-x1, player[i].position[1]-y1]).reshape(1,2)/(player[i].h_d(obs_1.position, r1)+r1)
            #G_D_2=-np.array([player[i].position[0]-x2, player[i].position[1]-y2]).reshape(1,2)/(player[i].h_d(obs_2.position, r2)+r2)
            #q=np.add(-1*player[i].alpha*w2*G_2.reshape(2,)/v_max,-w3*G_D_1.reshape(2,))
            #q=np.add(q,-w3*G_D_2.reshape(2,))
            q=-1*player[i].alpha*w2*G_2.reshape(2,)/v_max
            #h=np.vstack((h_2)).reshape(1,)
            h=np.vstack((h_1,h_2,h_3,h_4,h_5)).reshape(5,)
            u=quadprog_solve_qp(P, q, G, h, A=None, b=None)
            
             
            if counter==0:
                current_pose[i]= player[i].position
            elif counter==50:
                counter=0
                if np.linalg.norm(np.array(current_pose[i]-player[i].position)) < 20:
                   player[i].weight=player[i].weight*0.05
                   print(i)
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
                player[i].speed=(u[0]**2+u[1]**2)**0.1
        counter=counter+1  
        for i in range(len(player)):  
            playersprite[i].update()
            player[i].alphadot=(len(player)*player[i].weight-weight_sum)*beta
        
        obs_playersprite_1.update() 
        obs_playersprite_2.update() 
        obs_playersprite_3.update()
            
        screen.fill((255,255,255))  
        for i in range(len(out_player)):
            out_playersprite[i].draw(screen)   
        for i in range(len(player)):
            playersprite[i].draw(screen) 
        obs_playersprite_1.draw(screen)
        obs_playersprite_2.draw(screen)      
        obs_playersprite_3.draw(screen)    
        
        radius=int(v_max*(t_max-time.time()+t0))
        pg.draw.circle(screen, (0,0,0), (x0,y0), radius,1)
        pg.draw.circle(screen, (0,0,0), (x0,y0), r0,1)
        pg.draw.circle(screen, (200,200,200), (x4,y4), r4,0)
        pg.draw.circle(screen, (200,200,200), (x5,y5), r5,0)
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
            if test <= 0.01 or player[i].h_c(time.time()-t0)<=0:
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
    plt.ylabel('\u03B1') 
    plt.title('\u03B1 - t') 
    plt.legend(loc="upper right") 
    plt.show() 
    
 
if __name__ == '__main__':
    
    #set the initial pose for the robots
    initial_pose=[[230,380],[50,80],[450,100]]
    rgb=[(0,102,255),(255,153,0),(0,153,0)]
    main(initial_pose,rgb)
    while (True):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
    
