from tqdm import tqdm

def count():
    res = 0
    for i in tqdm(range(10000000)):
        res += i * i
    return res

if __name__ == "__main__":
    fim = count()

    print(f'Valor final: {fim}')
