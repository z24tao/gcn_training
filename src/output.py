from src.compress import *
from src.data import *


def to_hex(x, precision, line_len):
    linear = np.flip(x.flatten() if len(x.shape) > 1 else x)
    result = [np.binary_repr(val, width=precision) for val in linear]
    result = bin_to_hex(result)
    result = row_invert(result, line_len)
    return result


def bin_to_hex(b):
    bin_str = ''.join(b)
    return hex(int(bin_str, 2))[2:].zfill(len(bin_str) // 4)


def row_invert(data, line_len):
    result = ''
    if len(data) % line_len != 0:
        data = ''.join(['0' for _ in range(line_len - (len(data) % line_len))]) + data
    for start in range(0, len(data), line_len):
        result = data[start:start + line_len] + result
    return result


def preprocess(epoch, x, ws, bs, e, g, tile_size=128, row_blk_size=16, weight_partition=8, precision=16, line_len=128):
    slr_div = [0, 3, 5, 8]  # assigning workload to (Xilinx) super logic regions
    fs = ['' for _ in slr_div]  # file data

    # file address, header length, body length
    fa, hl, bl = [[] for _ in slr_div], [[] for _ in slr_div[:-1]], [[] for _ in slr_div[:-1]]

    x_tiles = coo_tile(mat_to_coo(x), slr_div[-1], tile_size)
    e_tiles = coo_tile(mat_to_coo(e), slr_div[-1], tile_size)

    for slr, pe_start, pe_end in zip(range(len(slr_div)-1), slr_div[:-1], slr_div[1:]):
        for pe_data in x_tiles[pe_start:pe_end]:
            for x_tile in pe_data:
                [tile_header, tile_col, tile_val] = coo_to_pcoo(x_tile, row_blk_size, weight_partition)
                tile_addr, tile_header_len, tile_body_len = len(fs[slr]) * 4, len(tile_header), len(tile_col)
                fs[slr] += to_hex(tile_val, precision, line_len)
                fa[slr].append(tile_addr)
                hl[slr].append(tile_header_len)
                bl[slr].append(tile_body_len)

        for pe_data in e_tiles[pe_start:pe_end]:
            for e_tile in pe_data:
                [tile_header, tile_col, tile_val] = coo_to_pcoo(e_tile, row_blk_size, weight_partition)
                tile_addr, tile_header_len, tile_body_len = len(fs[slr]) * 4, len(tile_header), len(tile_col)
                fs[slr] += to_hex(tile_header, 3, line_len)
                fs[slr] += to_hex(tile_col, precision, line_len)
                fs[slr] += to_hex(tile_val, precision, line_len)
                fa[slr].append(tile_addr)
                hl[slr].append(tile_header_len)
                bl[slr].append(tile_body_len)

    for w in ws:
        fa[-1].append(len(fs[-1]) * 4)
        fs[-1] += to_hex(w, precision, line_len)

    for b in bs:
        fa[-1].append(len(fs[-1]) * 4)
        fs[-1] += to_hex(b, precision * 2, line_len)

    fa[-1].append(len(fs[-1]) * 4)
    fs[-1] += to_hex(g, precision, line_len)

    write_files(fs, epoch)
    write_addrs(fa, hl, bl)


def write_files(fs, epoch):
    for i, data in enumerate(fs):
        write_file(data, f'/content/epoch{epoch}_file{i+1}')


def write_addrs(fa, hl, bl):
    writer = csv.writer(open('addrs.txt', 'w'), delimiter=',')
    writer.writerow(['addr', 'header_length', 'body_length'])
    for f in range(len(hl)):
        writer.writerow([f'file {f + 1}'])
        for h, c, v in zip(fa[f], hl[f], bl[f]):
            writer.writerow([h, c, v])
    writer.writerow([f'file {len(fa)}'])
    for h in fa[-1]:
        writer.writerow([h])
