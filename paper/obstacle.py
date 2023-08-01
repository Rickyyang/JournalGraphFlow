import math
import pygame as pg
import numpy as np
from pygame.math import Vector2
import time
import quadprog
import matplotlib.pyplot as plt
import random

r0=50
x0=250
y0=250

r_d=30
x_d=220
y_d=350

t_max=23
v_max=15
d=40
deta=2

w1=0.2
w2=8
w3=0.01

w_alpha=0.03

labda=1.5

def theta_fi(theta,fi):
    if abs(theta-fi)<=180:
        result=theta-fi
    elif theta>fi:
        result=theta-fi-360
    else:
        result=theta-fi+360
    return result

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

    def h_c(self,t):
        return t_max-t-distance(self.position)/v_max  
    
    def h_d(self, pos):
        return ((self.position[0]- pos[0])**2+(self.position[1]- pos[1])**2)**0.5-r_d

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
   
def main():
    pg.init()

    screen = pg.display.set_mode((500, 500))
    screen.fill((255,255,255))
    initial_pose=[[30,350],[210,380],[130,120],[400,300],[400,180]]

    length=len(initial_pose)  
  
    player = []
    playersprite=[]
    out_player = []
    out_playersprite = []

    for idx in range(len(initial_pose)):
        
        player.append(Player((initial_pose[idx][0],initial_pose[idx][1])))
        playersprite.append(pg.sprite.RenderPlain((player[idx])))

    obs=Player((x_d,y_d),rgb=(200,200,200),radius=r_d, obs=True)
    obs_playersprite=pg.sprite.RenderPlain((obs))
    obs.angle=60
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
        print(player[i].alpha)
    clock = pg.time.Clock()
    done = False
    t0=time.time()
    
    points=[[],[],[],[],[]]
    a=[[],[],[],[],[]]
    out_a=[]
    t=[]
    out_points = []
    while not done:
        
        clock.tick(60)  
        t1=time.time()-t0
        #print("time")
        #print(t1)
        weight_sum=0
        t.append(t1)
        for i in range(len(player)):  
            points[i].append((player[i].position[0],player[i].position[1]))
            
            a[i].append(player[i].alpha)
            P=w1*np.identity(2)
            
            G=np.array([player[i].position[0]-x0, player[i].position[1]-y0]).reshape(1,2)
            G_D=-np.array([player[i].position[0]-obs.position[0], player[i].position[1]-obs.position[1]]).reshape(1,2)/(player[i].h_d(obs.position)+r_d)
            h_D=np.array(player[i].h_d(obs.position))
            q=-1*player[i].alpha*w2*G.reshape(2,)*np.array([player[i].position[0]-x0, player[i].position[1]-y0]).reshape(1,2)/(v_max*(distance(player[i].position)-r0))
            h=np.array([(-1+player[i].h_c(t1))*v_max])
            
            G_new=np.vstack((G,G_D))
            h_new=0.001*np.vstack((h,h_D)).reshape(2,)
            player[i].weight= player[i].alpha*player[i].h_c(t1)    
            weight_sum = weight_sum + player[i].weight
            u=quadprog_solve_qp(P, q, G_new, h_new, A=None, b=None)
            #cvxopt_solve_qp(P, q, G, h, A=None, b=None)
            #print(i)
            print(u)
            delete_idx=[]
            if u[0]**2+u[1]**2 >= 2:
                player[i].speed=0
                player[i].angle_speed=0
                out_player.append(player[i])
                out_playersprite.append(playersprite[i])
                out_points.append(points[i])
                out_a.append(a[i])
                delete_idx.append(i)
            else:
                player[i].angle_speed=-1*labda*(theta_fi(player[i].angle, player[i].fi(-u[1],-u[0]))) 
                player[i].speed=(u[0]**2+u[1]**2)**0.5
           
        for i in range(len(player)):  
            playersprite[i].update()
            player[i].alphadot=(len(player)*player[i].weight-weight_sum)*w_alpha
            #print(player[i].alphadot)
        obs.speed=0.1
        obs.angle_speed=0
        obs_playersprite.update() 
         
        screen.fill((255,255,255))  
        for i in range(len(out_player)):
            out_playersprite[i].draw(screen)   
        for i in range(len(player)):
            playersprite[i].draw(screen)         
        obs_playersprite.draw(screen)
        radius=int(v_max*(t_max-time.time()+t0))
        
        pg.draw.circle(screen, (0,0,0), (x0,y0), radius,1)
        pg.draw.circle(screen, (0,0,0), (x0,y0), r0,1)
        pg.display.flip()
         
        buchang=0
        delta_t=time.time()-t1-t0
        #print('delta t')
        #print(delta_t)
        
        for i in range(len(player)):
            print(player[i].alpha)
            test=player[i].alpha+player[i].alphadot*delta_t
            
            if test <= 0.05 or player[i].h_c(time.time()-t0)<=0:
                player[i].speed=0
                player[i].angle_speed=0
                buchang=buchang+player[i].alpha
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
        Closed=False    
        for i in range(len(player)):
            player[i].alpha+=buchang/len(player)
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
        print(len(out_a[i]))
        plt.plot(t,out_a[i],label='robot_'+str(index))
        SUM=np.add(SUM,out_a[i]).tolist()
        index+=1
    for i in range(len(a)):
        a[i].extend([0]*(len(t)-len(a[i])))
        
        plt.plot(t,a[i],label='robot_'+str(index))
        SUM=np.add(SUM,a[i]).tolist()
        index+=1
    plt.plot(t,SUM,label='sum')
    # naming the x axis 
    plt.xlabel('t - axis') 
    # naming the y axis 
    plt.ylabel('\u03B1 - axis') 
    # giving a title to my graph 
    plt.title('\u03B1 - t graph') 
  
    # show a legend on the plot 
    plt.legend(loc="left") 
  
    # function to show the plot 
    plt.show() 
    
 
if __name__ == '__main__':
    main()
    while (True):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
    
