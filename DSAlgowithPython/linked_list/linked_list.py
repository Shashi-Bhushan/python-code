#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>


class Node:
    def __init__(self, data, prev = None, next = None):
        self.data = data
        self.prev = prev
        self.next = next

    def has_next(self):
        return self.next is not None

    def has_prev(self):
        return self.prev is not None


class LinkedList(object):
    """Single Linked List implementation."""

    def __init__(self, head=None, length: int = 0):
        self.head = head
        self.length = length

    def insert_at_beginning(self, node: Node) -> None:
        node.next = self.head
        self.head = node
        self.length += 1

    def insert_at_index(self, node: Node, index: int) -> None:
        if index <= self.length:
            current = self.head
            current_index = 1

            while current_index != index:
                current = current.next
                current_index += 1

            node.next = current.next
            current.next = node

            self.length += 1

    def insert_at_end(self, node: Node) -> None:
        self.insert_at_index(node, self.length)

    def delete_from_beginning(self):
        self.head = self.head.next
        self.length -= 1

    def delete_from_index(self, index):
        current = self.head
        current_index = 1

        while current_index != index:
            current = current.next
            current_index += 1

        current.next = current.next.next
        self.length -= 1

    def delete_from_end(self):
        self.delete_from_index(self.length - 1)

    def __repr__(self):
        string = ''

        current = self.head

        while current is not None:
            string += f'({current.data}), '
            current = current.next

        return string


if __name__ == '__main__':
    linked_list = LinkedList()

    linked_list.insert_at_beginning(Node(5))
    print(linked_list)
    linked_list.insert_at_beginning(Node(4))
    print(linked_list)
    linked_list.insert_at_beginning(Node(6))
    print(linked_list)
    linked_list.insert_at_index(Node(16), 1)
    print(linked_list)
    linked_list.insert_at_index(Node(12), 2)
    print(linked_list)
    linked_list.insert_at_end(Node(10))
    print(linked_list)
    linked_list.delete_from_index(3)
    print(linked_list)
    linked_list.delete_from_end()
    print(linked_list)
