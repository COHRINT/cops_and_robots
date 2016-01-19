from __future__ import division
import logging
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import box, Polygon
from shapely.affinity import scale

from cops_and_robots.fusion.softmax import Softmax, SoftmaxClass, BinarySoftmax


def _make_regular_2D_poly(n_sides=3, origin=(0, 0), theta=0, max_r=1):
    x = [max_r * np.cos(2 * np.pi * n / n_sides + theta) + origin[0]
         for n in range(n_sides)]
    y = [max_r * np.sin(2 * np.pi * n / n_sides + theta) + origin[1]
         for n in range(n_sides)]
    pts = zip(x,y)

    logging.debug('Generated points:\n{}'.format('\n'.join(map(str,pts))))

    return Polygon(pts)


def camera_model_2D(min_view_dist=0., max_view_dist=2):
    """Generate a softmax model for a two dimensional camera.

    A softmax model decomposing the space into five subclasses: four
    `no-detection` subclasses, and one `detection` subclass. This uses the
    Microsoft Kinect viewcone specifications of 57 degrees viewing angle.

    Classes are numbered as such:
    <>TODO: insert image of what it looks like

    Parameters
    ----------
    min_view_dist : float, optional
        The minimum view distance for the camera. Note that this is not a hard
        boundary, and specifies the distance at which an object has a 5% chance
        of detection. The default it 0 meters, which implies no minimum view
        distance.
    min_view_dist : float, optional
        The minimum view distance for the camera. Note that this is not a hard
        boundary, and specifies the distance at which an object has a 5% chance
        of detection. The default is 1 meter.

    """
    # Define view cone centered at origin
    horizontal_view_angle = 57  # from Kinect specifications (degrees)
    min_y = min_view_dist * np.tan(np.radians(horizontal_view_angle / 2))
    max_y = max_view_dist * np.tan(np.radians(horizontal_view_angle / 2))
    camera_poly = Polygon([(min_view_dist,  min_y),
                           (max_view_dist,  max_y),
                           (max_view_dist, -max_y),
                           (min_view_dist, -min_y)])

    if min_view_dist > 0:
        steepness = [100, 100, 2.5, 100, 100]
        labels = ['Detection', 'No Detection', 'No Detection', 'No Detection',
                  'No Detection']
    else:
        steepness = [100, 100, 2.5, 100,]
        labels = ['Detection', 'No Detection', 'No Detection', 'No Detection']

    bounds = [-11.25, -3.75, 3.75, 3.75]
    camera = Softmax(poly=camera_poly, labels=labels,
                     steepness=steepness, bounds=bounds)
    return camera


def speed_model():
    """Generate a one-dimensional Softmax model for speeds.
    """
    labels = ['Stopped', 'Slow', 'Medium', 'Fast']
    sm = Softmax(weights=np.array([[0], [150], [175], [200]]),
                 biases=np.array([0, -2.5, -6, -14]),
                 state_spec='x', labels=labels,
                 bounds=[0, 0, 0.4, 0.4])
    return sm


def binary_speed_model():
    sm = speed_model()
    bsm = BinarySoftmax(sm)
    return bsm

def speed_model_2d(num_classes=7, bounds=[-2, -2, 2, 2], spread=1.3):
    """Generate a one-dimensional Softmax model for speeds.
    """
    # labels = ['Stopped', 'Slow', 'Medium', 'Fast']
    # rs = [0, 0.2, 0.4, 0.8]
    labels = ['Stopped', 'Slow', 'Fast']
    rs = [0, 0.3, 0.6]
    sms = []

    for i, label in enumerate(labels[1:], start=1):
        try:
            poly = scale(poly, 1 + rs[i], 1 + rs[i])
        except:
            poly = _make_regular_2D_poly(num_classes - 1, max_r=rs[i])

        class_labels = [labels[0]] + [label] * (num_classes - 1)
        steepnesses = [10 + i * 20] * num_classes
        sm = Softmax(poly=poly, labels=class_labels,
                     steepness=steepnesses, bounds=bounds)

        if i == 1:
            full_sm = sm
        else:
            new_weights = sm.weights[1:]
            new_biases = sm.biases[1:] - spread ** i
            new_labels = [label] * (num_classes - 1)
            new_steepnesses = steepnesses[1:]

            full_sm.add_classes(new_weights, new_biases, new_labels, 
                                new_steepnesses, poly=poly)

    return full_sm


def binary_speed_model_2d():
    sm = speed_model_2d()
    bsm = BinarySoftmax(sm)
    return bsm


def pentagon_model():
    poly = _make_regular_2D_poly(5, max_r=2, theta=np.pi/3.1)
    labels = ['Interior',
              'Mall Terrace Entrance',
              'Heliport Facade',
              'South Parking Entrance', 
              'Concourse Entrance',
              'River Terrace Entrance', 
             ]
    steepness = 5
    sm = Softmax(poly=poly, labels=labels, resolution=0.1, steepness=5)
    return sm


def range_model(poly=None, spread=1, bounds=None):
    if poly == None:
        poly = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    num_classes = len(poly.exterior.coords)
    labels = ['Inside'] + ['Near'] * (num_classes - 1)
    steepnesses = [10] * num_classes
    sm = Softmax(poly=poly, labels=labels, resolution=0.1,
                 steepness=steepnesses, bounds=bounds)

    steepnesses = [11] * num_classes
    # far_bounds = _make_regular_2D_poly(4, max_r=3, theta=np.pi/4)
    larger_poly = scale(poly, 2, 2)
    labels = ['Inside'] + ['Outside'] * (num_classes - 1)
    sm_far = Softmax(poly=poly, labels=labels, resolution=0.1,
                 steepness=steepnesses, bounds=bounds)

    new_weights = sm_far.weights[1:]
    new_biases = sm_far.biases[1:] - spread
    new_labels = ['Outside'] * (num_classes - 1)
    new_steepnesses = steepnesses[1:]

    sm.add_classes(new_weights, new_biases, new_labels, new_steepnesses)
    return sm


def binary_range_model(poly=None, bounds=None):
    dsm = range_model(poly, bounds=bounds)
    bdsm = BinarySoftmax(dsm, bounds=bounds)
    return bdsm


def intrinsic_space_model(poly=None, bounds=None):
    if poly == None:
        poly = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)


    # <>TODO: If sides != 4, find a way to make it work!
    # NOTE: front and back are intrinsic, left and right are extrinsic
    labels = ['Inside', 'Front', 'Right', 'Back', 'Left']
    steepness = 3
    sm = Softmax(poly=poly, labels=labels, resolution=0.1,
                 steepness=steepness, bounds=bounds)
    return sm


def binary_intrinsic_space_model(poly=None, bounds=None, allowed_relations=None, 
                                 container_poly=None):
    if bounds is None:
        bounds = [-5, -5, 5, 5]
    ism = intrinsic_space_model(poly, bounds=bounds)

    if container_poly is not None:
        n, o = normals_from_polygon(container_poly)

        steepness = 10
        outside_weights = n * steepness + ism.weights[1:] 
        outside_biases = o * steepness + ism.biases[1:] 

        labels = ['Outside'] * 4
        # labels = ['Outside_Front','Outside_Left','Outside_Back','Outside_Right']
        ism.add_classes(outside_weights, outside_biases, labels)

    
    # <>TODO: remove this debug stub
    # axes = ism.plot(plot_poly=True)
    # patch = PolygonPatch(container_poly, facecolor='white', zorder=5,
    #                      linewidth=5, edgecolor='brown',)
    # axes[0].add_patch(patch)
    # plt.show()

    bism = BinarySoftmax(ism, bounds=bounds)
    return bism


def demo_models():
    logging.info('Preparing Softmax models for demo...')
    # Regular Softmax models #################################################

    # Speed model
    sm = speed_model()
    title='Softmax Speed Model'
    logging.info('Building {}'.format(title))
    sm.plot(show_plot=False, plot_3D=False, title=title)

    # Pentagon Model
    pm = pentagon_model()
    title='Softmax Area Model (The Pentagon)'
    logging.info('Building {}'.format(title))
    pm.plot(show_plot=False, title=title)

    # Range model
    x = [-2, 0, 2, 2]
    y = [-3, -1, -1, -3]
    pts = zip(x,y)
    poly = Polygon(pts)
    rm = range_model(poly)
    # rm = intrinsic_space_model(poly)
    title='Softmax Intrinsic Space Model (Irregular)'
    logging.info('Building {}'.format(title))
    rm.plot(show_plot=False, title=title)

    # Multimodal softmax models ##############################################

    # Range model (irregular poly)
    x = [-2, 0, 2, 2]
    y = [-3, -1, -1, -3]
    pts = zip(x,y)
    poly = Polygon(pts)
    rm = range_model(poly)
    title='MMS Range Model (Irregular)'
    logging.info('Building {}'.format(title))
    rm.plot(show_plot=False, title=title)

    # Camera model (with movement)
    cm = camera_model_2D()
    cm.move([-2,-2, 90])
    title='MMS Camera Model'
    logging.info('Building {}'.format(title))
    cm.plot(show_plot=False, title=title)
    cm.move([-5,-2, 90])
    title='MMS Camera Model (moved, detect only)'
    logging.info('Building {}'.format(title))
    cm.plot(show_plot=False, class_='Detection', title=title)

    # Binary softmax models ##################################################

    # Binary speed model 
    bsm = binary_speed_model()
    title='Binary MMS Speed Model'
    logging.info('Building {}'.format(title))
    bsm.binary_models['Medium'].plot(show_plot=False, title=title)

    # Binary Pentagon Model - Individual and multiple plots
    pent = pentagon_model()
    bpent = BinarySoftmax(pent)
    title='Binary MMS Pentagon Model (Not Heliport only)'
    logging.info('Building {}'.format(title))
    bpent.binary_models['Heliport Facade'].plot(show_plot=False, class_='Not Heliport Facade',
                                                title=title)
    title='Binary MMS Pentagon Model (Interior)'
    bpent.binary_models['Interior'].plot(show_plot=False, title=title)

    # Binary Range Model
    bdsm = binary_range_model()
    title='Binary MMS Range Model'
    logging.info('Building {}'.format(title))
    bdsm.binary_models['Near'].plot(show_plot=False, title=title)

    # Binary intrinsic space model
    bism = binary_intrinsic_space_model()
    title='Binary MMS Intrinsic Space Model'
    logging.info('Building {}'.format(title))
    bism.binary_models['Front'].plot(show_plot=False, title=title)
    plt.show()

if __name__ == '__main__':
    # a = 2  # half-width
    # b = 1  # horizontal inset
    # c = 1  # vertical inset
    # d = 3  # half-height
    # x = [-a, -a, -b, -b, -a, -a, a, a, b, b, a, a, -a]
    # y = [-d, -c, -c,  c,  c,  d, d, c, c, -c, -c, -d, -d]
    # y = [-0.58, -0.58, 0.58, 0.58, -0.58]
    # x = [-0.35, 0.35, 0.35, -0.35, -0.35]
    # pose = [-8.24, -2.15, 270]
    # x = [a + pose[0] for a in x]
    # y = [a + pose[1] for a in y]
    
    # pts = zip(x,y)
    # poly = Polygon(pts)
    # bounds = [-9.5, -3.33, 4, 3.68]
    # rm = range_model(poly, bounds=bounds)
    sm = speed_model_2d()
    bsm = binary_speed_model_2d()

    title=''
    sm.plot(title=title, plot_poly=True)
    bsm.binary_models['Medium'].plot(title=title)