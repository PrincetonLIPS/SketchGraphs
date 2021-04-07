"""Functions for drawing sketches using matplotlib

This module implements local plotting functionality in order to render Onshape sketches using matplotlib.

"""

import math

import numpy as np
import numpy.linalg as npla
import numpy.random as npr

import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt

from ._entity import Arc, Circle, Line, Point

def matern32(x, ls, amp):
  x2d = np.atleast_2d(x)/ls
  sqdists = (x2d-x2d.T)**2
  dists = np.sqrt(sqdists)
  K = (1 + np.sqrt(3)*dists)*np.exp(-np.sqrt(3)*dists) + 1e-6*np.eye(x2d.shape[0])
  return amp**2 * K

def matern52(x, ls, amp):
  x2d = np.atleast_2d(x)/ls
  sqdists = (x2d-x2d.T)**2
  dists = np.sqrt(sqdists)
  K = (1 + np.sqrt(5)*dists + 5*sqdists/3)*np.exp(-np.sqrt(5)*dists) + 1e-6*np.eye(x2d.shape[0])
  return amp**2 * K

class Noisifier:
  def __init__(self, sketch, resolution=500):
    self.sketch = sketch
    self.resolution = resolution

    self._get_ranges()
    self._get_chol()

  def _get_ranges(self):
    self.min_x = np.inf
    self.max_x = -np.inf
    self.min_y = np.inf
    self.max_y = -np.inf
    def update_x(x):
      if x > self.max_x:
        self.max_x = x
      elif x < self.min_x:
        self.min_x = x
    def update_y(y):
      if y > self.max_y:
        self.max_y = y
      elif y < self.min_y:
        self.min_y = y

    for ent in self.sketch.entities.values():
      if isinstance(ent, (Arc, Circle)):
        update_x(-ent.radius)
        update_x(ent.radius)
        update_y(-ent.radius)
        update_y(ent.radius)
        update_x(ent.xCenter)
        update_y(ent.yCenter)
      elif isinstance(ent, Line):
        update_x(ent.start_point[0])
        update_x(ent.end_point[0])
        update_y(ent.start_point[1])
        update_y(ent.end_point[1])
      elif isinstance(ent, Point):
        update_x(ent.x)
        update_y(ent.y)

  def _get_chol(self, ls=0.05, amp=0.002):
    self.scale = 3*np.sqrt((self.max_x-self.min_x)**2 + (self.max_y-self.min_y)**2)
    self.x = np.linspace(0, 1, self.resolution)
    K = matern32(self.x, ls, amp)
    self.cK = npla.cholesky(K)

  def get_line(self, start_x, start_y, end_x, end_y):
    length = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
    max_idx = int(np.floor((length / self.scale) * self.resolution))

    y = self.scale * self.cK[:max_idx,:max_idx] @ npr.randn(max_idx)
    x = self.x[:max_idx] * self.scale

    theta = np.arctan2(end_y-start_y, end_x-start_x)
    newx = start_x + x * np.cos(theta) - y * np.sin(theta)
    newy = start_y + y * np.cos(theta) + x * np.sin(theta)

    return newx, newy

  def get_arc(self, center_x, center_y, radius, start, end):
    start = np.pi * start / 180
    end = np.pi * end / 180
    length = radius * (end-start)
    max_idx = int(np.floor((length / self.scale) * self.resolution))

    y = self.scale * self.cK[:max_idx,:max_idx] @ npr.randn(max_idx)

    thetas = np.linspace(start, end, max_idx)
    newx = center_x + (radius + y) * np.cos(thetas)
    newy = center_y + (radius + y) * np.sin(thetas)

    return newx, newy

  def get_circle(self, center_x, center_y, radius):
    gap = npr.rand() * 360
    return self.get_arc(center_x, center_y, radius, gap, gap+359)

  def get_point(self, center_x, center_y):
    x, y = self.get_circle(0, 0, self.scale*0.1)
    x *= 0.01
    y *= 0.01
    return x + center_x, y + center_y

def _get_linestyle(entity):
    return '--' if entity.isConstruction else '-'

def sketch_point(ax, point: Point, color='black', show_subnodes=False, noisy=None):
  if noisy:
    x, y = noisy.get_point(point.x, point.y)
    ax.plot(x, y, color=color, linewidth=1)
  else:
    ax.scatter(point.x, point.y, c=color, marker='.')

def sketch_line(ax, line: Line, color='black', show_subnodes=False, noisy=None):
    start_x, start_y = line.start_point
    end_x, end_y = line.end_point
    if show_subnodes:
        marker = '.'
    else:
        marker = None
    if noisy:
      x, y = noisy.get_line(start_x, start_y, end_x, end_y)

      ax.plot(x, y, color, linestyle=_get_linestyle(line), linewidth=1, marker=marker)
    else:
      ax.plot((start_x, end_x), (start_y, end_y), color, linestyle=_get_linestyle(line), linewidth=1, marker=marker)

def sketch_circle(ax, circle: Circle, color='black', show_subnodes=False, noisy=None):

    if noisy:
      x, y = noisy.get_circle(circle.xCenter, circle.yCenter, circle.radius)
      ax.plot(x, y, color=color, linewidth=1)
    else:
      patch = matplotlib.patches.Circle(
        (circle.xCenter, circle.yCenter), circle.radius,
        fill=False, linestyle=_get_linestyle(circle), color=color)

      ax.add_patch(patch)
    if show_subnodes:
      ax.scatter(circle.xCenter, circle.yCenter, c=color, marker='.', zorder=20)

def sketch_arc(ax, arc: Arc, color='black', show_subnodes=False, noisy=None):
    angle = math.atan2(arc.yDir, arc.xDir) * 180 / math.pi
    startParam = arc.startParam * 180 / math.pi
    endParam = arc.endParam * 180 / math.pi

    if arc.clockwise:
        startParam, endParam = -endParam, -startParam

    if noisy:
      x, y = noisy.get_arc(arc.xCenter, arc.yCenter, arc.radius, startParam, endParam)
      ax.plot(x, y, color, linewidth=1)
    else:
      ax.add_patch(
        matplotlib.patches.Arc(
          (arc.xCenter, arc.yCenter), 2*arc.radius, 2*arc.radius,
          angle=angle, theta1=startParam, theta2=endParam,
          linestyle=_get_linestyle(arc), color=color))

    if show_subnodes:
        ax.scatter(arc.xCenter, arc.yCenter, c=color, marker='.')
        ax.scatter(*arc.start_point, c=color, marker='.', zorder=40)
        ax.scatter(*arc.end_point, c=color, marker='.', zorder=40)


_PLOT_BY_TYPE = {
    Arc: sketch_arc,
    Circle: sketch_circle,
    Line: sketch_line,
    Point: sketch_point
}


def render_sketch(sketch, ax=None, show_axes=False, show_origin=False, hand_drawn=False, show_subnodes=False):
    """Renders the given sketch using matplotlib.

    Parameters
    ----------
    sketch : Sketch
        The sketch instance to render
    ax : matplotlib.Axis, optional
        Axis object on which to render the sketch. If None, a new figure is created.
    show_axes : bool
        Indicates whether axis lines should be drawn
    show_origin : bool
        Indicates whether origin point should be drawn
    hand_drawn : bool
        Indicates whether to emulate a hand-drawn appearance
    show_subnodes : bool
        Indicates whether endpoints/centerpoints should be drawn

    Returns
    -------
    matplotlib.Figure
        If `ax` is not provided, the newly created figure. Otherwise, `None`.
    """
    noisy = None
    if hand_drawn:
        saved_rc = mpl.rcParams.copy()
        noisy = Noisifier(sketch)
        #plt.xkcd(scale=1, length=100, randomness=3)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        fig = None

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    if not show_axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        _ = [line.set_marker('None') for line in ax.get_xticklines()]
        _ = [line.set_marker('None') for line in ax.get_yticklines()]

        # Eliminate lower and left axes
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')

    if show_origin:
        point_size = mpl.rcParams['lines.markersize'] * 1
        ax.scatter(0, 0, s=point_size, c='black')

    for ent in sketch.entities.values():
        sketch_fn = _PLOT_BY_TYPE.get(type(ent))
        if sketch_fn is None:
            continue
        sketch_fn(ax, ent, show_subnodes=show_subnodes, noisy=noisy)

    # Rescale axis limits
    ax.relim()
    ax.autoscale_view()

    if hand_drawn:
        mpl.rcParams.update(saved_rc)

    return fig


def render_graph(graph, filename):
    """Renders the given pgv.AGraph to an image file.

    Parameters
    ----------
    graph : pgv.AGraph
        The graph to render
    filename : string
        Where to save the image file

    Returns
    -------
    None
    """
    graph.layout('dot')
    graph.draw(filename)


__all__ = ['render_sketch', 'render_graph']
