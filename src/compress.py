import numpy as np


def mat_to_coo(data):
    result = []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                result.append([i, j, val])
    return np.array(result)


def coo_tile(coo, pe, tile_size):
    density = [0 for _ in range(pe)]
    num_rows, num_cols = np.max(coo[:, 0]) + 1, np.max(coo[:, 1]) + 1
    tiles = [[[] for _ in range(0, num_cols, tile_size)] for _ in range(pe)]
    for [r, c, v] in coo:
        density[r % pe] += 1
        tiles[r % pe][c // tile_size].append([r // pe, c, v])
    return [[np.array(tile) for tile in row] for row in tiles]


# row_blk_size: number of non-zeros processed simultaneously
# weight_partition: number of partitions in distributed ram, where bank collision cannot occur across collision
def coo_to_pcoo(coo, row_blk_size, weight_partition):
    num_rows, num_cols = np.max(coo[:, 0]) + 1, np.max(coo[:, 1]) + 1
    sor, eor, vld = 4, 2, 1

    csr = [[] for _ in range(num_rows)]
    for [r, c, v] in coo:
        csr[r].append([1, c, v])  # 1 for elem header (sor=? eor=? vld=1)

    blk = [[] for _ in range(row_blk_size)]
    for r, elems in enumerate(csr):
        if len(elems) == 0:
            blk[r % row_blk_size].append([sor + eor, 0, 0])
        else:
            elems[0][0] += sor
            elems[-1][0] += eor
            blk[r % row_blk_size].extend(elems)
    blk = resolve_collision(blk, weight_partition=weight_partition)
    return header_body_split(blk)


def resolve_collision(pcoo, weight_partition=128):
    row_blk_size = len(pcoo)
    out, assigned, total = [], 0, sum([len(row) for row in pcoo])
    reading_row = 0
    read_pos = [0 for _ in range(row_blk_size)]
    out_used = [-1 for _ in range(weight_partition)]
    out_row = [[] for _ in range(row_blk_size)]
    out_row_assigned = 0

    while assigned < total:
        if not out_row[reading_row]:
            reading_col = read_pos[reading_row]
            if reading_col < len(pcoo[reading_row]):
                [header, col, val] = pcoo[reading_row][reading_col]
                col_partition = col % weight_partition
                if header % 2 == 0:
                    out_row[reading_row] = [header, col, val]
                    read_pos[reading_row] += 1
                    assigned += 1
                elif out_used[col_partition] in [-1, col]:
                    out_row[reading_row] = [header, col, val]
                    out_used[col_partition] = col
                    read_pos[reading_row] += 1
                    assigned += 1
                else:
                    out_row[reading_row] = [0, 0, 0]
            else:
                out_row[reading_row] = [0, 0, 0]
            out_row_assigned += 1

            if out_row_assigned == row_blk_size:
                out.append(out_row)
                out_used = [-1 for _ in range(weight_partition)]
                out_row = [[] for _ in range(row_blk_size)]
                out_row_assigned = 0

        reading_row = (reading_row + 1) % row_blk_size

    empty = True
    for i in range(row_blk_size):
        if out_row[i] is None:
            out_row[i] = [0, 0, 0]
        else:
            empty = False

    if not empty:
        out.append(out_row)

    for i in range(len(out) - 1, -1, -1):
        empty = True
        for [header, _, _] in out[i]:
            if header % 2 != 0:
                empty = False
                break
        if empty:
            out = out[:-1]

    return out


# split [sor + eor + vld, col, val] into [sor + eor + vld], [col], [val]
def header_body_split(pcoo):
    header = []
    cols = [[] for _ in range(len(pcoo[0]))]
    vals = [[] for _ in range(len(pcoo[0]))]
    for row in pcoo:
        header.append([elem[0] for elem in row])
        for pe, elem in enumerate(row):
            if elem[0] % 2 == 1:
                cols[pe].append(elem[1])
                vals[pe].append(elem[2])
    longest = max([len(pe) for pe in cols])
    for i, pe in enumerate(cols):
        pe_length = len(pe)
        if pe_length < longest:
            cols[i].extend([0 for _ in range(longest - pe_length)])
            vals[i].extend([0 for _ in range(longest - pe_length)])
    cols = np.transpose(np.array(cols))
    vals = np.transpose(np.array(vals))
    return np.array(header), cols, vals
