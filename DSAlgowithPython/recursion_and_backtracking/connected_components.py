#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

"""
Problem Statement: Finding the maximum length of connected cells of 1s (regions) in a matrix of 0s and 1s.
"""

from pprint import pprint


class Node:
    """Node represents an individual item in a matrix

    Each node has two attributes
        val: value represented by the Node. For a binary array, it's 1s and 0s
        position: a tuple having Node coordinates in (row, column) format
    """

    def __init__(self, val: int, position: tuple):
        self.val = val
        self.position = position

    def neighbours(self, matrix) -> list:
        """Return all the nodes adjacent to self in the matrix

        Valid adjacent directions are
                Up

            Left    Right

               Down
        """
        nodes: list[Node] = []

        row, column = self.position[0], self.position[1]

        # Left Bound Check
        if row - 1 >= 0:
            node = matrix[row - 1][column]
            nodes.append(node)

        # Right Bound Check
        if row + 1 < len(matrix[0]):
            node = matrix[row + 1][column]
            nodes.append(node)

        # Upper Bound Check
        if column - 1 >= 0:
            node = matrix[row][column - 1]
            nodes.append(node)

        # Lower Bound Check
        if column + 1 < len(matrix):
            node = matrix[row][column + 1]
            nodes.append(node)

        return nodes

    def same_value_neighbours(self, matrix: list) -> list:
        """Returns the nodes with the same value as self, in self's neighbors in a given matrix"""
        return [node for node in self.neighbours(matrix) if node.val == self.val]

    def __repr__(self):
        return f'{self.position} -> {self.val}'

    def __hash__(self):
        return hash((self.val, *self.position))

    def __eq__(self, other) -> bool:
        """Compare Node objects on basis on Position and value"""
        if isinstance(other, self.__class__):
            return self.val == other.val and self.position == other.position
        else:
            return False

    @staticmethod
    def to_node_matrix(int_matrix: list) -> list:
        """expects a matrix on integers and returns a matrix of corresponding Node objects"""
        node_matrix = []

        for x, row in enumerate(int_matrix):
            node_matrix.append([Node(val=item, position=(x, y)) for y, item in enumerate(row)])

        return node_matrix


def get_all_connected_components(node: Node, matrix: list, visited_matrix: set = None) -> list:
    """Returns all connected components with same value recursively

    :returns
        a list of all same valued nodes connected to this node

    Algorithm:
    - Create a set for tracking Visited nodes or use existing one
      Intention is when function is called while traversing the node matrix, visited nodes set will be none.
      a new set will be created here. All the nodes which are 'visited' has already been processed.
    - Add current node to visited nodes' matrix.
    - Get all neighbours of the node with same value
    - If neighbour is not already visited(processed)
        - Get all connected components of the neighbour node and add to Nodes

    """
    visited_matrix: set = visited_matrix or set()
    visited_matrix.add(node)

    nodes: list = []

    for neighbour in node.same_value_neighbours(matrix):
        if neighbour not in visited_matrix:
            nodes.append(neighbour)
            nodes += get_all_connected_components(neighbour, matrix, visited_matrix)

    return nodes


if __name__ == '__main__':
    matrix = Node.to_node_matrix([
        [1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1]
    ])

    max_connected_components = 0

    for row in matrix:
        for node in row:
            if node.val != 0:
                same_value_nodes = get_all_connected_components(node, matrix)
                same_value_nodes.append(node)

                max_connected_components = len(same_value_nodes) if max_connected_components < len(same_value_nodes) \
                    else max_connected_components

    print('Node Matrix is ')
    pprint(matrix)
    print(f'Maximum number of connected components : {max_connected_components}')
