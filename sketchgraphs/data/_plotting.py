"""Functions for drawing sketches using matplotlib

This module implements local plotting functionality in order to render Onshape sketches using matplotlib.

"""

import math
from contextlib import nullcontext

import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt

from ._entity import Arc, Circle, Line, Point


def _get_linestyle(entity):
    return '--' if entity.isConstruction else '-'

def sketch_point(ax, point: Point, color='black', show_subnodes=False):
    ax.scatter(point.x, point.y, c=color, marker='.')

def sketch_line(ax, line: Line, color='black', show_subnodes=False):
    start_x, start_y = line.start_point
    end_x, end_y = line.end_point
    if show_subnodes:
        marker = '.'
    else:
        marker = None
    ax.plot((start_x, end_x), (start_y, end_y), color, linestyle=_get_linestyle(line), linewidth=1, marker=marker)

def sketch_circle(ax, circle: Circle, color='black', show_subnodes=False):
    patch = matplotlib.patches.Circle(
        (circle.xCenter, circle.yCenter), circle.radius,
        fill=False, linestyle=_get_linestyle(circle), color=color)
    if show_subnodes:
        ax.scatter(circle.xCenter, circle.yCenter, c=color, marker='.', zorder=20)
    ax.add_patch(patch)

def sketch_arc(ax, arc: Arc, color='black', show_subnodes=False):
    angle = math.atan2(arc.yDir, arc.xDir) * 180 / math.pi
    startParam = arc.startParam * 180 / math.pi
    endParam = arc.endParam * 180 / math.pi

    if arc.clockwise:
        startParam, endParam = -endParam, -startParam

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


def render_sketch(sketch, ax=None, show_axes=False, show_origin=False,
                  hand_drawn=False, show_subnodes=False, show_points=True):
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
    show_points : bool
        Indicates whether Point entities should be drawn

    Returns
    -------
    matplotlib.Figure
        If `ax` is not provided, the newly created figure. Otherwise, `None`.
    """
    with plt.xkcd(
        scale=1, length=100, randomness=3) if hand_drawn else nullcontext():

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
            if isinstance(ent, Point):
                if not show_points:
                    continue
            if sketch_fn is None:
                continue
            sketch_fn(ax, ent, show_subnodes=show_subnodes)

        # Rescale axis limits
        ax.relim()
        ax.autoscale_view()

        return fig


def render_graph(graph, filename, show_node_idxs=False):
    """Renders the given pgv.AGraph to an image file.

    Parameters
    ----------
    graph : pgv.AGraph
        The graph to render
    filename : string
        Where to save the image file
    show_node_idxs: bool
        If true, append node indexes to their labels


    Returns
    -------
    None
    """
    if show_node_idxs:
        for idx, node in enumerate(graph.nodes()):
            node.attr['label'] += ' (' + str(idx) + ')'
    graph.layout('dot')
    graph.draw(filename)


__all__ = ['render_sketch', 'render_graph']