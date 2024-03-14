import json
import os
import sqlite3

DB_PATH = 'barren/data.db'
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database path '{DB_PATH}' does not exist.")


def initialize_database(table: str):
    """
    Create the database, if data.db does not exist.
    Then check whether the specified table exists, and create the table if it does not exist.
    """
    try:
        if not isinstance(table, str):
            raise TypeError(f"table:{table} must be a string")
    except TypeError as e:
        print(e)

    if table == 'single':
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS single (
            modified BOOLEAN,
            qubit INTEGER,
            layer INTEGER,
            paras TEXT,
            gradients TEXT,
            variance REAL,
            UNIQUE(modified, qubit, layer)
        )
        '''
    elif table == 'multi':
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS multi (
            modified BOOLEAN,
            qubit INTEGER,
            layer INTEGER,
            gradients TEXT,
            variance REAL,
            UNIQUE(modified, qubit, layer)
        )
        '''
    else:
        assert False, f"Unknown table name: {table}."

    # 连接到SQLite数据库，如果数据库不存在，会自动创建
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()
    print(f"Table '{table}' is ready in database '{DB_PATH}'.")


def table_exists(table: str) -> bool:
    """Check whether the table exists"""
    try:
        if not isinstance(table, str):
            raise TypeError(f"table:{table} must be a string")
    except TypeError as e:
        print(e)

    if not os.path.exists(DB_PATH):
        initialize_database(table)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def save_data(data: dict, table: str):
    """
    Save the data into 'barren/data.db'.

    table = 'single':
    param data: A list of dictionaries. len(paras)=len(gradients)=samples.
            data = {'modified': bool, 'qubit': int, 'layer': int, 'paras': list[float], 'gradients': list[float], 'variance': float}.

    table = 'multi'
    param data: A list of dictionaries. len(gradients)=samples.
            data = {'modified': bool, 'qubit': int, 'layer': int, 'gradients': list[float], 'variance': float}.
    """
    try:
        if not isinstance(data, dict):
            raise TypeError(f"data:{data} must be a dictionary")
        if not isinstance(table, str):
            raise TypeError(f"table:{table} must be a string")
    except TypeError as e:
        print(e)
    if not os.path.exists(DB_PATH) or not table_exists(table):
        initialize_database(table)

    if table == 'single':
        save_data_sql = ('''INSERT OR REPLACE INTO single (modified, qubit, layer, paras, gradients, variance) VALUES (?, ?, ?, ?, ?, ?)''',
                         (data['modified'], data['qubit'], data['layer'], json.dumps(data['paras']), json.dumps(data['gradients']), data['variance']))
    elif table == 'multi':
        save_data_sql = ('''INSERT OR REPLACE INTO multi (modified, qubit, layer, gradients, variance) VALUES (?, ?, ?, ?, ?)''',
                         (data['modified'], data['qubit'], data['layer'], json.dumps(data['gradients']), data['variance']))
    else:
        assert False, f"Unknown table name: {table}."

    sql_statement, paras = save_data_sql
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute(sql_statement, paras)
    db.commit()
    db.close()
