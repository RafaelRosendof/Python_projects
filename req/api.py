import requests , json


url = "https://jsonplaceholder.typicode.com/todos/1"

def get_some():

    request = requests.get(url)

    print("get something:\n\n ")
    print("status code response", request.status_code)
    print("\n\n")
    print("content response", request.content)
    print("\n\n")
    todos = json.loads(request.content)
    print(todos)

    
def get_todos():
    request = requests.get(url)

    print("get todos:\n\n ")
    print("status code response" , request.status_code)
    print("\n")
    print("content response" , request.content)
    todos = json.loads(request.content)
    print(todos)


def teste1():
    params = {'q': 'Python' , 'limit': '10'}
    response = requests.get("https://api.exemplo.com/posts", params=params)
    print(response.url)
    

def del_teste():
    request = requests.delete("https://jsonplaceholder.typicode.com/todos/1")
    print("delete todos:\n\n ")
    print(request.status_code)

def post():
    json = {
        'title' : 'figas',
        'body' : 'teste',
        'userId' : 1
    }

    request = requests.post("https://jsonplaceholder.typicode.com/todos" , json)
    print("post todos:\n\n ")
    print("status code response" , request.status_code)
    print("content response" , request.content)


def main():
    print("\n\n\n\nIniciando Consumo de API \n\n\n\n\n")

    get_some()

    print("\n\n")

    get_todos()

    print("\n\n")

    #teste1()

    del_teste()

    print("\n\n")

    post()


if __name__ == "__main__":
    main()