class Queries:

    create_user_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL,
            password TEXT NOT NULL
        );
    """

    create_conversation_table = """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            date TEXT NOT NULL,
            FOREIGN KEY(userId) REFERENCES users(id)
        );
    """

    insert_user = """
        INSERT INTO users 
        (username, role, password) 
        VALUES (?, ?, ?);
    """
    
    get_user_by_id = """
        SELECT * 
        FROM users 
        WHERE id = ?;
    """

    get_user_by_username = """
        SELECT * 
        FROM users 
        WHERE username = ?;
    """

    insert_chat = """
        INSERT INTO conversations 
        (userId, prompt, response, date) 
        VALUES (?, ?, ?, ?);
    """

    fetch_all_chat = """
        SELECT * 
        FROM conversations 
        WHERE userId = ?;
    """

    get_chat_by_id = """
        SELECT * 
        FROM conversations
        WHERE id =?;
    """
