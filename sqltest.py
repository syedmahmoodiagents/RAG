from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

DATABASE_URL = "sqlite:///company.db"

engine = create_engine(DATABASE_URL)

db = SQLDatabase(engine)

print(db.get_usable_table_names())
print(db.get_table_info())
print(db.get_table_info(["customers"]))
print(db.run("SELECT * FROM customers;"))
