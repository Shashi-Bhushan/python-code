#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>


class Node:
    def __init__(self, data, prev, next):
        """
        Packaged Linked list node representation using only one pointer for forward and backward operations
        position is actually stored as pointer difference, where ptrDiff = position of previous node ^ position of next node

        :param data: the data contained on this node
        :param prev: the prev node reference
        :param next: the next node reference
        """
        self.data = data
        self.position = prev ^ next

    def has_next(self):
        return self.next is not None

    def has_prev(self):
        return self.prev is not None


class PackedLinkedList:
    def __init__(self, head: Node = None, length: int = 0):
        self.head = head
        self.length = length

    def insert_at_beginning(self, node):
        node.next = self.head
        self.head = node
        self.length += 1


