class User:
    name = ""
    email = ""
    age = 0

user = User()
user.name = "Satish"
user.email = "satish@gmail.com"
user.age = "My age is 22"

# print(user)



# Using pydantic
from pydantic import BaseModel, Field
from typing import List, Annotated

class User(BaseModel):
    name: str
    email: str
    age: int

user = User(name="Satish", email="satish@gmail.com", age=22)

# print(user)


# field validation
class Product(BaseModel):
    title: str = Field(max_length=20)
    desc: str = Field(min_length=20)
    price: int = Field(gt=0, le=100)

product = Product(title="Samsung Galaxy", desc="This is amazing mobile phone from samsung", price=50)
product2 = Product(title="Iphone 17", desc="This is amazing mobile phone from apple", price=99)
print(product)


class Order(BaseModel):
    id: str = Field(min_length=5, max_length=20)
    product: Product

order = Order(id="lskjdfuslkdjf", product=product)
print(order, order.product.title)


class Carts(BaseModel):
    id: str = Field(min_length=5, max_length=20)
    items: List[Product]
    comment: str = Field(default="No Comments")

cart = Carts(id="lsdjflsjdfl", items=[product, product2])
print(cart)

for product in cart.items:
    print(product.title)



# Annotated
commonValidator = Annotated(List[Product], Field(default=[]))

