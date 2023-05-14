import struct

# apre il file di testo in lettura
with open('input.txt', 'r') as f:
    # legge i numeri come stringhe e li converte in interi
    numbers = [int(x) for x in f.read().split(" ")]

# apre il file binario in scrittura
with open('output.bin', 'wb') as f:
    # scrive ogni numero come una sequenza di 4 byte in formato big-endian
    for n in numbers:
        f.write(struct.pack('>i', n))