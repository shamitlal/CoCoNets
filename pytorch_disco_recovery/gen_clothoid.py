import numpy as np
from scipy.special import fresnel
import pylab
import matplotlib.pyplot as plt
import utils_py

# # t = np.linspace(-10, 10, 1000)
# t = np.linspace(0, 10, 1000)
# ssa, csa = fresnel(t)

# alpha = 30.0 # rand from 6 to 80
# accel = 0.01 # rand from -5 to 5

initial_x = 0.0
initial_y = 0.0
initial_vel = 0.1 # distance per time unit

# noyaw
tangent = np.array([1, 0]).reshape(2, 1)
normal = np.array([0, 1]).reshape(2, 1)

# i think this is close enough, and next i need to deploy this and see that
# my trajectories/velocities make sense in carla
# theta_range = np.linspace(-0.1, 0.1, 3)
# print('theta_range', theta_range)
# theta_range = [-.1, 0., .1]
# print('theta_range', theta_range)

offset_range = np.linspace(0, 0.4, 5)
print('offset_range', offset_range)

theta_range = [0]
print('theta_range', theta_range)
# offset_range = [0]
# print('offset_range', offset_range)

# for y_coeff in [-1, 1]:
for y_coeff in [1]:
    for alpha in np.linspace(6, 80, 10):
        print('alpha = %d' % alpha)
        for accel in [0]:
            # for accel in np.linspace(-0.1, 0.1, 5):

            # let's pred 20 frames, 
            # at 3fps,
            # meaning 6.66 seconds of time

            # time = np.linspace(0, 6.66, 20)
            # time = np.linspace(0, 6.66, 100)
            # time = np.linspace(0, 3.0, 100)
            # vel_profile = initial_vel + time*accel

            # dist = np.zeros_like(vel_profile)
            dist = np.zeros(100)
            for t in list(range(1, 100)):
                # dist[t] = dist[t-1] + vel_profile[t] + accel
                dist[t] = dist[t-1] + initial_vel + t*accel
            
            # print('time\n', time)
            # print('vel_profile\n', vel_profile)
            # print('dist\n', dist)
            # print('dist/alpha\n', dist/alpha)

            for theta in theta_range:
                for offset in offset_range:
                    r0 = np.reshape(np.array([np.cos(theta), -np.sin(theta)]), (1, 2))
                    r1 = np.reshape(np.array([np.sin(theta), np.cos(theta)]), (1, 2))
                    rot = np.concatenate([r0,r1], axis=0)

                    ssa, csa = fresnel(offset+dist/alpha)
                    curve = alpha * ((csa.T*tangent).T + (ssa.T*normal).T)

                    curve = (np.dot(rot, curve.T)).T

                    curve[:,1] *= y_coeff

                    curve = curve - curve[0]

                    # # # o = *fresnel(t)

                    # # print(ssa[:10])
                    # # print(csa[:10])

                    # # plt.scatter(ssa, csa, c='k.')
                    # plt.scatter(ssa, csa, s=0.5)
                    plt.scatter(curve[:,0], curve[:,1], s=0.25)
                    # plt.hold(True)

## straight lines
for y_coeff in [-1, 1]:
    for accel in [0]:
        dist = np.zeros(100)
        for t in list(range(1, 100)):
            # dist[t] = dist[t-1] + vel_profile[t] + accel
            dist[t] = dist[t-1] + initial_vel + t*accel
        for theta in theta_range:
            for offset in offset_range:
                r0 = np.reshape(np.array([np.cos(theta), -np.sin(theta)]), (1, 2))
                r1 = np.reshape(np.array([np.sin(theta), np.cos(theta)]), (1, 2))
                rot = np.concatenate([r0,r1], axis=0)

                # x = dist
                # y = np.zeros_like(x)
                curve = np.zeros((100, 2))
                curve[:,0] = dist

                # curve = alpha * ((.T*tangent).T + (ssa.T*normal).T)

                curve = (np.dot(rot, curve.T)).T

                curve[:,1] *= y_coeff

                plt.scatter(curve[:,0], curve[:,1], s=0.25)

# for radius in np.linspace(1.0, 3, 5):
#     theta = np.linspace(0, 2*np.pi, 100)
#     # radius = np.sqrt(0.6)
#     x = radius*np.cos(theta)
#     y = radius*np.sin(theta) + radius
#     plt.scatter(x, y, s=0.5)

plt.axis('equal')
plt.savefig('clothoid_samples.png')


# i think i
# # pylab.plot(*fresnel(t), c='k')
# # pylab.plot(*fresnel(t), c='k')
# # pylab.savefig('foo.png')

# # pylab.show()

