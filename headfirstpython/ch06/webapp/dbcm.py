#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1 @ outlook dot com>

"""The UseDatabase context manager for working with MySQL/MariaDB.

Simple example usage:

    from DBcm import UseDatabase, SQLError

    config = { 'host': '127.0.0.1',
               'user': 'myUserid',
               'password': 'myPassword',
               'database': 'myDB' }

    with UseDatabase(config) as cursor:
        try:
            _SQL = "select * from log"
            cursor.execute(_SQL)
            data = cursor.fetchall()
        except SQLError as err:
            print('Your query broke:', str(err))

Enjoy, and have fun.
"""

##############################################################################
# Context manager for connecting/disconnecting to a database.
##############################################################################

import mysql.connector


class DBConnectionError(Exception):
    """Raised if the backend database cannot be connected to"""
    pass


class CredentialsError(Exception):
    """Raised if the database credentials are not valid"""
    pass


class SQLError(Exception):
    """Raised if the sql query caused problem"""
    pass


class UseDatabase:
    def __init__(self, config):
        """Add the database configuration parameters to the object.

        This class expects a single dictionary argument which needs to assign
        the appropriate values to (at least) the following keys:

            host - the IP address of the host running MySQL/MariaDB.
            user - the MySQL/MariaDB username to use.
            password - the user's password.
            database - the name of the database to use.

        For more options, refer to the mysql-connector-python documentation.
        """
        self.config = config

    def __enter__(self):
        """Connect to database and create a DB cursor.

        Return the database cursor to the context manager.
        Raise ConnnectionError if the database can't be found.
        Raise CredentialsError if the wrong username/password used.
        """
        try:
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor()
            return self.cursor
        except mysql.connector.errors.InterfaceError as err:
            raise ConnectionError(err) from err
        except mysql.connector.errors.ProgrammingError as err:
            raise CredentialsError(err) from err

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the cursor as well as the connection (after committing).

        Raise ProgrammingError as an SQLError, and re-raise anything else.
        """
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

        if exc_type is mysql.connector.errors.ProgrammingError:
            raise SQLError(exc_val)
        elif exc_type:
            raise exc_type(exc_val)
