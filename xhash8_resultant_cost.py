#!/usr/bin/env python3
"""Direct computation of the structured XHash8 resultant.

Target:
    Res_z(z**7 - g, a[0] + a[1]*z + ... + a[6]*z**6).

The computation uses 227 general ring multiplications, including
9 squarings. It uses only additions, multiplications, and
multiplications by fixed integer constants.

For quotient rings, the multiplication callback performs the
required reductions.

Examples:
    python xhash7_resultant_227_clean.py
    python xhash7_resultant_227_clean.py --random-tests 100 --quotient-tests 5
"""
from __future__ import annotations

import argparse
import json
import operator
from collections import Counter, defaultdict
from itertools import permutations
from random import Random
from typing import Callable, Sequence, TypeVar

P = 2**64 - 2**32 + 1
T = TypeVar("T")
# Inputs v[0],...,v[6] = a[0],...,a[6]; v[7] = g.
# Instruction k stores its result at v[8+k].
# ("mul", i, j): v[i]*v[j].
# ("add", i, j): v[i]+v[j].
# ("scale", i, c): c*v[i], for a fixed integer c.
PROGRAM: tuple[tuple[str, int, int], ...] = (
    ('mul', 6, 6),  # v[8]
    ('mul', 7, 8),  # v[9]
    ('mul', 8, 9),  # v[10]
    ('mul', 6, 8),  # v[11]
    ('mul', 10, 11),  # v[12]
    ('mul', 5, 5),  # v[13]
    ('mul', 4, 6),  # v[14]
    ('scale', 14, -1),  # v[15]
    ('add', 13, 15),  # v[16]
    ('mul', 8, 16),  # v[17]
    ('mul', 1, 8),  # v[18]
    ('mul', 17, 18),  # v[19]
    ('scale', 19, 7),  # v[20]
    ('scale', 14, -7),  # v[21]
    ('add', 13, 21),  # v[22]
    ('mul', 5, 22),  # v[23]
    ('mul', 3, 8),  # v[24]
    ('scale', 24, 7),  # v[25]
    ('add', 23, 25),  # v[26]
    ('mul', 5, 26),  # v[27]
    ('mul', 2, 6),  # v[28]
    ('scale', 28, -1),  # v[29]
    ('mul', 4, 4),  # v[30]
    ('scale', 30, 2),  # v[31]
    ('add', 29, 31),  # v[32]
    ('mul', 8, 32),  # v[33]
    ('scale', 33, 7),  # v[34]
    ('add', 27, 34),  # v[35]
    ('mul', 5, 35),  # v[36]
    ('mul', 14, 24),  # v[37]
    ('scale', 37, -21),  # v[38]
    ('add', 36, 38),  # v[39]
    ('mul', 5, 39),  # v[40]
    ('mul', 0, 6),  # v[41]
    ('mul', 3, 3),  # v[42]
    ('mul', 2, 4),  # v[43]
    ('scale', 43, 2),  # v[44]
    ('add', 42, 44),  # v[45]
    ('scale', 45, -1),  # v[46]
    ('add', 41, 46),  # v[47]
    ('mul', 6, 47),  # v[48]
    ('mul', 4, 30),  # v[49]
    ('add', 48, 49),  # v[50]
    ('mul', 11, 50),  # v[51]
    ('scale', 51, -7),  # v[52]
    ('add', 40, 52),  # v[53]
    ('mul', 5, 53),  # v[54]
    ('add', 29, 30),  # v[55]
    ('mul', 8, 55),  # v[56]
    ('mul', 24, 56),  # v[57]
    ('scale', 57, 7),  # v[58]
    ('add', 54, 58),  # v[59]
    ('add', 20, 59),  # v[60]
    ('add', 12, 60),  # v[61]
    ('mul', 7, 61),  # v[62]
    ('mul', 1, 6),  # v[63]
    ('mul', 5, 16),  # v[64]
    ('scale', 24, -3),  # v[65]
    ('add', 64, 65),  # v[66]
    ('mul', 5, 66),  # v[67]
    ('add', 28, 31),  # v[68]
    ('mul', 8, 68),  # v[69]
    ('add', 67, 69),  # v[70]
    ('mul', 63, 70),  # v[71]
    ('scale', 71, 7),  # v[72]
    ('mul', 2, 5),  # v[73]
    ('mul', 3, 4),  # v[74]
    ('scale', 74, -2),  # v[75]
    ('add', 73, 75),  # v[76]
    ('mul', 5, 76),  # v[77]
    ('scale', 41, 3),  # v[78]
    ('scale', 42, 3),  # v[79]
    ('scale', 43, -1),  # v[80]
    ('add', 79, 80),  # v[81]
    ('add', 78, 81),  # v[82]
    ('mul', 6, 82),  # v[83]
    ('add', 49, 83),  # v[84]
    ('add', 77, 84),  # v[85]
    ('mul', 5, 85),  # v[86]
    ('scale', 28, -5),  # v[87]
    ('add', 31, 87),  # v[88]
    ('mul', 3, 6),  # v[89]
    ('mul', 88, 89),  # v[90]
    ('add', 86, 90),  # v[91]
    ('mul', 5, 91),  # v[92]
    ('mul', 2, 2),  # v[93]
    ('scale', 93, 3),  # v[94]
    ('mul', 0, 4),  # v[95]
    ('scale', 95, -1),  # v[96]
    ('add', 94, 96),  # v[97]
    ('mul', 6, 97),  # v[98]
    ('scale', 98, -1),  # v[99]
    ('scale', 42, 5),  # v[100]
    ('scale', 43, -2),  # v[101]
    ('add', 100, 101),  # v[102]
    ('mul', 4, 102),  # v[103]
    ('add', 99, 103),  # v[104]
    ('mul', 6, 104),  # v[105]
    ('mul', 30, 30),  # v[106]
    ('scale', 106, 2),  # v[107]
    ('add', 105, 107),  # v[108]
    ('mul', 6, 108),  # v[109]
    ('scale', 109, -1),  # v[110]
    ('add', 92, 110),  # v[111]
    ('mul', 5, 111),  # v[112]
    ('scale', 41, -2),  # v[113]
    ('add', 42, 80),  # v[114]
    ('add', 113, 114),  # v[115]
    ('mul', 6, 115),  # v[116]
    ('scale', 49, 3),  # v[117]
    ('add', 116, 117),  # v[118]
    ('mul', 24, 118),  # v[119]
    ('add', 112, 119),  # v[120]
    ('scale', 120, -7),  # v[121]
    ('add', 72, 121),  # v[122]
    ('mul', 1, 122),  # v[123]
    ('mul', 0, 5),  # v[124]
    ('mul', 3, 124),  # v[125]
    ('scale', 125, -7),  # v[126]
    ('mul', 0, 2),  # v[127]
    ('mul', 6, 127),  # v[128]
    ('scale', 128, 2),  # v[129]
    ('mul', 2, 42),  # v[130]
    ('add', 93, 95),  # v[131]
    ('mul', 4, 131),  # v[132]
    ('add', 130, 132),  # v[133]
    ('add', 129, 133),  # v[134]
    ('scale', 134, 7),  # v[135]
    ('add', 126, 135),  # v[136]
    ('mul', 5, 136),  # v[137]
    ('scale', 43, 3),  # v[138]
    ('add', 42, 138),  # v[139]
    ('mul', 4, 139),  # v[140]
    ('add', 98, 140),  # v[141]
    ('mul', 3, 141),  # v[142]
    ('scale', 142, -7),  # v[143]
    ('add', 137, 143),  # v[144]
    ('mul', 5, 144),  # v[145]
    ('mul', 0, 0),  # v[146]
    ('mul', 6, 146),  # v[147]
    ('scale', 147, 2),  # v[148]
    ('mul', 0, 42),  # v[149]
    ('scale', 149, -1),  # v[150]
    ('add', 93, 96),  # v[151]
    ('mul', 2, 151),  # v[152]
    ('scale', 152, 2),  # v[153]
    ('add', 150, 153),  # v[154]
    ('add', 148, 154),  # v[155]
    ('mul', 6, 155),  # v[156]
    ('scale', 43, 5),  # v[157]
    ('add', 42, 157),  # v[158]
    ('mul', 42, 158),  # v[159]
    ('scale', 95, 3),  # v[160]
    ('add', 93, 160),  # v[161]
    ('mul', 30, 161),  # v[162]
    ('scale', 162, -1),  # v[163]
    ('add', 159, 163),  # v[164]
    ('add', 156, 164),  # v[165]
    ('mul', 6, 165),  # v[166]
    ('scale', 42, 2),  # v[167]
    ('add', 43, 167),  # v[168]
    ('mul', 49, 168),  # v[169]
    ('add', 166, 169),  # v[170]
    ('scale', 170, 7),  # v[171]
    ('add', 145, 171),  # v[172]
    ('mul', 5, 172),  # v[173]
    ('scale', 128, -1),  # v[174]
    ('scale', 130, 3),  # v[175]
    ('scale', 93, 2),  # v[176]
    ('scale', 95, -5),  # v[177]
    ('add', 176, 177),  # v[178]
    ('mul', 4, 178),  # v[179]
    ('add', 175, 179),  # v[180]
    ('add', 174, 180),  # v[181]
    ('mul', 6, 181),  # v[182]
    ('mul', 30, 81),  # v[183]
    ('add', 182, 183),  # v[184]
    ('mul', 6, 184),  # v[185]
    ('mul', 30, 49),  # v[186]
    ('add', 185, 186),  # v[187]
    ('mul', 3, 187),  # v[188]
    ('scale', 188, -7),  # v[189]
    ('add', 173, 189),  # v[190]
    ('mul', 5, 190),  # v[191]
    ('mul', 41, 131),  # v[192]
    ('scale', 192, 7),  # v[193]
    ('scale', 95, -3),  # v[194]
    ('add', 176, 194),  # v[195]
    ('mul', 42, 195),  # v[196]
    ('mul', 43, 161),  # v[197]
    ('scale', 197, -1),  # v[198]
    ('add', 196, 198),  # v[199]
    ('scale', 199, 7),  # v[200]
    ('add', 193, 200),  # v[201]
    ('mul', 6, 201),  # v[202]
    ('mul', 42, 114),  # v[203]
    ('add', 95, 176),  # v[204]
    ('mul', 30, 204),  # v[205]
    ('add', 203, 205),  # v[206]
    ('mul', 4, 206),  # v[207]
    ('scale', 207, 7),  # v[208]
    ('add', 202, 208),  # v[209]
    ('mul', 6, 209),  # v[210]
    ('mul', 106, 114),  # v[211]
    ('scale', 211, 7),  # v[212]
    ('add', 210, 212),  # v[213]
    ('mul', 6, 213),  # v[214]
    ('mul', 49, 106),  # v[215]
    ('add', 214, 215),  # v[216]
    ('add', 191, 216),  # v[217]
    ('add', 123, 217),  # v[218]
    ('add', 62, 218),  # v[219]
    ('mul', 7, 219),  # v[220]
    ('mul', 5, 18),  # v[221]
    ('scale', 221, 7),  # v[222]
    ('mul', 3, 5),  # v[223]
    ('scale', 223, -1),  # v[224]
    ('scale', 28, -3),  # v[225]
    ('add', 31, 225),  # v[226]
    ('add', 224, 226),  # v[227]
    ('mul', 5, 227),  # v[228]
    ('mul', 3, 14),  # v[229]
    ('add', 228, 229),  # v[230]
    ('mul', 5, 230),  # v[231]
    ('scale', 43, -3),  # v[232]
    ('add', 167, 232),  # v[233]
    ('scale', 233, -1),  # v[234]
    ('add', 41, 234),  # v[235]
    ('mul', 6, 235),  # v[236]
    ('add', 49, 236),  # v[237]
    ('mul', 6, 237),  # v[238]
    ('scale', 238, -1),  # v[239]
    ('add', 231, 239),  # v[240]
    ('scale', 240, 7),  # v[241]
    ('add', 222, 241),  # v[242]
    ('mul', 1, 242),  # v[243]
    ('mul', 5, 195),  # v[244]
    ('scale', 41, 5),  # v[245]
    ('scale', 114, 2),  # v[246]
    ('add', 245, 246),  # v[247]
    ('mul', 3, 247),  # v[248]
    ('add', 244, 248),  # v[249]
    ('mul', 5, 249),  # v[250]
    ('scale', 128, -5),  # v[251]
    ('scale', 130, 2),  # v[252]
    ('scale', 93, 5),  # v[253]
    ('scale', 95, -2),  # v[254]
    ('add', 253, 254),  # v[255]
    ('mul', 4, 255),  # v[256]
    ('scale', 256, -1),  # v[257]
    ('add', 252, 257),  # v[258]
    ('add', 251, 258),  # v[259]
    ('mul', 6, 259),  # v[260]
    ('mul', 30, 139),  # v[261]
    ('add', 260, 261),  # v[262]
    ('scale', 262, -1),  # v[263]
    ('add', 250, 263),  # v[264]
    ('mul', 5, 264),  # v[265]
    ('scale', 95, 2),  # v[266]
    ('add', 93, 266),  # v[267]
    ('mul', 6, 267),  # v[268]
    ('scale', 268, -1),  # v[269]
    ('scale', 43, -5),  # v[270]
    ('add', 79, 270),  # v[271]
    ('mul', 4, 271),  # v[272]
    ('scale', 272, -1),  # v[273]
    ('add', 269, 273),  # v[274]
    ('mul', 6, 274),  # v[275]
    ('add', 106, 275),  # v[276]
    ('mul', 3, 276),  # v[277]
    ('add', 265, 277),  # v[278]
    ('scale', 278, 7),  # v[279]
    ('add', 243, 279),  # v[280]
    ('mul', 1, 280),  # v[281]
    ('mul', 5, 146),  # v[282]
    ('mul', 3, 127),  # v[283]
    ('add', 282, 283),  # v[284]
    ('mul', 5, 284),  # v[285]
    ('mul', 41, 178),  # v[286]
    ('mul', 42, 267),  # v[287]
    ('add', 94, 177),  # v[288]
    ('mul', 43, 288),  # v[289]
    ('add', 287, 289),  # v[290]
    ('add', 286, 290),  # v[291]
    ('scale', 291, -1),  # v[292]
    ('add', 285, 292),  # v[293]
    ('mul', 5, 293),  # v[294]
    ('scale', 95, -15),  # v[295]
    ('add', 93, 295),  # v[296]
    ('mul', 2, 296),  # v[297]
    ('add', 149, 297),  # v[298]
    ('scale', 298, -1),  # v[299]
    ('add', 148, 299),  # v[300]
    ('mul', 6, 300),  # v[301]
    ('add', 95, 253),  # v[302]
    ('mul', 30, 302),  # v[303]
    ('scale', 303, -1),  # v[304]
    ('add', 203, 304),  # v[305]
    ('add', 301, 305),  # v[306]
    ('mul', 3, 306),  # v[307]
    ('scale', 307, -1),  # v[308]
    ('add', 294, 308),  # v[309]
    ('mul', 5, 309),  # v[310]
    ('mul', 2, 147),  # v[311]
    ('scale', 311, -3),  # v[312]
    ('mul', 42, 127),  # v[313]
    ('scale', 313, -2),  # v[314]
    ('scale', 95, 5),  # v[315]
    ('add', 93, 315),  # v[316]
    ('mul', 93, 316),  # v[317]
    ('mul', 30, 146),  # v[318]
    ('scale', 318, -1),  # v[319]
    ('add', 317, 319),  # v[320]
    ('add', 314, 320),  # v[321]
    ('add', 312, 321),  # v[322]
    ('mul', 6, 322),  # v[323]
    ('scale', 179, -1),  # v[324]
    ('add', 252, 324),  # v[325]
    ('mul', 42, 325),  # v[326]
    ('mul', 2, 97),  # v[327]
    ('mul', 30, 327),  # v[328]
    ('scale', 328, -1),  # v[329]
    ('add', 326, 329),  # v[330]
    ('add', 323, 330),  # v[331]
    ('mul', 6, 331),  # v[332]
    ('add', 42, 232),  # v[333]
    ('mul', 42, 333),  # v[334]
    ('mul', 30, 151),  # v[335]
    ('add', 334, 335),  # v[336]
    ('mul', 30, 336),  # v[337]
    ('add', 332, 337),  # v[338]
    ('add', 310, 338),  # v[339]
    ('scale', 339, 7),  # v[340]
    ('add', 281, 340),  # v[341]
    ('mul', 1, 341),  # v[342]
    ('mul', 2, 161),  # v[343]
    ('scale', 149, -2),  # v[344]
    ('add', 343, 344),  # v[345]
    ('add', 147, 345),  # v[346]
    ('mul', 124, 346),  # v[347]
    ('scale', 347, -7),  # v[348]
    ('scale', 311, -2),  # v[349]
    ('scale', 313, -3),  # v[350]
    ('add', 320, 350),  # v[351]
    ('add', 349, 351),  # v[352]
    ('mul', 3, 352),  # v[353]
    ('scale', 353, 7),  # v[354]
    ('add', 348, 354),  # v[355]
    ('mul', 5, 355),  # v[356]
    ('mul', 147, 161),  # v[357]
    ('scale', 357, -1),  # v[358]
    ('mul', 149, 255),  # v[359]
    ('scale', 359, -1),  # v[360]
    ('mul', 93, 151),  # v[361]
    ('scale', 318, -5),  # v[362]
    ('add', 361, 362),  # v[363]
    ('mul', 2, 363),  # v[364]
    ('add', 360, 364),  # v[365]
    ('scale', 365, -1),  # v[366]
    ('add', 358, 366),  # v[367]
    ('mul', 6, 367),  # v[368]
    ('add', 94, 266),  # v[369]
    ('mul', 43, 369),  # v[370]
    ('scale', 370, -1),  # v[371]
    ('add', 287, 371),  # v[372]
    ('mul', 42, 372),  # v[373]
    ('add', 93, 194),  # v[374]
    ('mul', 93, 374),  # v[375]
    ('add', 318, 375),  # v[376]
    ('mul', 30, 376),  # v[377]
    ('add', 373, 377),  # v[378]
    ('add', 368, 378),  # v[379]
    ('scale', 379, 7),  # v[380]
    ('add', 356, 380),  # v[381]
    ('mul', 5, 381),  # v[382]
    ('mul', 0, 146),  # v[383]
    ('mul', 6, 383),  # v[384]
    ('scale', 384, -7),  # v[385]
    ('mul', 2, 288),  # v[386]
    ('add', 344, 386),  # v[387]
    ('mul', 0, 387),  # v[388]
    ('scale', 388, -7),  # v[389]
    ('add', 385, 389),  # v[390]
    ('mul', 6, 390),  # v[391]
    ('add', 149, 152),  # v[392]
    ('mul', 42, 392),  # v[393]
    ('add', 176, 254),  # v[394]
    ('mul', 93, 394),  # v[395]
    ('scale', 318, -3),  # v[396]
    ('add', 395, 396),  # v[397]
    ('mul', 4, 397),  # v[398]
    ('scale', 398, -1),  # v[399]
    ('add', 393, 399),  # v[400]
    ('scale', 400, -7),  # v[401]
    ('add', 391, 401),  # v[402]
    ('mul', 6, 402),  # v[403]
    ('scale', 43, -7),  # v[404]
    ('add', 42, 404),  # v[405]
    ('mul', 42, 405),  # v[406]
    ('add', 96, 176),  # v[407]
    ('mul', 30, 407),  # v[408]
    ('scale', 408, 7),  # v[409]
    ('add', 406, 409),  # v[410]
    ('mul', 42, 410),  # v[411]
    ('add', 93, 254),  # v[412]
    ('mul', 2, 412),  # v[413]
    ('mul', 49, 413),  # v[414]
    ('scale', 414, -7),  # v[415]
    ('add', 411, 415),  # v[416]
    ('add', 403, 416),  # v[417]
    ('mul', 3, 417),  # v[418]
    ('add', 382, 418),  # v[419]
    ('add', 342, 419),  # v[420]
    ('add', 220, 420),  # v[421]
    ('mul', 7, 421),  # v[422]
    ('mul', 4, 5),  # v[423]
    ('add', 89, 423),  # v[424]
    ('mul', 1, 424),  # v[425]
    ('scale', 425, -7),  # v[426]
    ('mul', 2, 3),  # v[427]
    ('scale', 427, 2),  # v[428]
    ('add', 124, 428),  # v[429]
    ('mul', 5, 429),  # v[430]
    ('add', 42, 43),  # v[431]
    ('mul', 4, 431),  # v[432]
    ('add', 268, 432),  # v[433]
    ('add', 430, 433),  # v[434]
    ('scale', 434, 7),  # v[435]
    ('add', 426, 435),  # v[436]
    ('mul', 1, 436),  # v[437]
    ('scale', 147, 3),  # v[438]
    ('scale', 149, 3),  # v[439]
    ('add', 152, 439),  # v[440]
    ('add', 438, 440),  # v[441]
    ('mul', 5, 441),  # v[442]
    ('scale', 132, 3),  # v[443]
    ('add', 130, 443),  # v[444]
    ('add', 174, 444),  # v[445]
    ('mul', 3, 445),  # v[446]
    ('add', 442, 446),  # v[447]
    ('scale', 447, -7),  # v[448]
    ('add', 437, 448),  # v[449]
    ('mul', 1, 449),  # v[450]
    ('mul', 2, 146),  # v[451]
    ('mul', 5, 451),  # v[452]
    ('scale', 452, -1),  # v[453]
    ('mul', 0, 3),  # v[454]
    ('mul', 178, 454),  # v[455]
    ('scale', 455, -1),  # v[456]
    ('add', 453, 456),  # v[457]
    ('mul', 5, 457),  # v[458]
    ('scale', 384, 2),  # v[459]
    ('mul', 2, 369),  # v[460]
    ('add', 149, 460),  # v[461]
    ('mul', 0, 461),  # v[462]
    ('scale', 462, -1),  # v[463]
    ('add', 459, 463),  # v[464]
    ('mul', 6, 464),  # v[465]
    ('add', 176, 315),  # v[466]
    ('mul', 2, 466),  # v[467]
    ('add', 149, 467),  # v[468]
    ('mul', 42, 468),  # v[469]
    ('scale', 318, 2),  # v[470]
    ('add', 361, 470),  # v[471]
    ('mul', 4, 471),  # v[472]
    ('add', 469, 472),  # v[473]
    ('add', 465, 473),  # v[474]
    ('add', 458, 474),  # v[475]
    ('scale', 475, 7),  # v[476]
    ('add', 450, 476),  # v[477]
    ('mul', 1, 477),  # v[478]
    ('mul', 5, 383),  # v[479]
    ('mul', 3, 479),  # v[480]
    ('scale', 480, 3),  # v[481]
    ('scale', 313, 5),  # v[482]
    ('add', 397, 482),  # v[483]
    ('add', 311, 483),  # v[484]
    ('mul', 0, 484),  # v[485]
    ('scale', 485, -1),  # v[486]
    ('add', 481, 486),  # v[487]
    ('mul', 5, 487),  # v[488]
    ('mul', 147, 302),  # v[489]
    ('scale', 489, -1),  # v[490]
    ('mul', 131, 149),  # v[491]
    ('scale', 491, 3),  # v[492]
    ('mul', 2, 471),  # v[493]
    ('add', 492, 493),  # v[494]
    ('add', 490, 494),  # v[495]
    ('mul', 3, 495),  # v[496]
    ('add', 488, 496),  # v[497]
    ('scale', 497, -7),  # v[498]
    ('add', 478, 498),  # v[499]
    ('mul', 1, 499),  # v[500]
    ('mul', 204, 479),  # v[501]
    ('scale', 501, 7),  # v[502]
    ('scale', 147, -2),  # v[503]
    ('add', 149, 327),  # v[504]
    ('add', 503, 504),  # v[505]
    ('mul', 3, 146),  # v[506]
    ('mul', 505, 506),  # v[507]
    ('scale', 507, -7),  # v[508]
    ('add', 502, 508),  # v[509]
    ('mul', 5, 509),  # v[510]
    ('mul', 147, 451),  # v[511]
    ('scale', 511, 7),  # v[512]
    ('add', 350, 376),  # v[513]
    ('mul', 146, 513),  # v[514]
    ('scale', 514, 7),  # v[515]
    ('add', 512, 515),  # v[516]
    ('mul', 6, 516),  # v[517]
    ('mul', 42, 451),  # v[518]
    ('scale', 518, 7),  # v[519]
    ('mul', 0, 471),  # v[520]
    ('scale', 520, 7),  # v[521]
    ('add', 519, 521),  # v[522]
    ('mul', 42, 522),  # v[523]
    ('scale', 95, -7),  # v[524]
    ('add', 93, 524),  # v[525]
    ('mul', 93, 525),  # v[526]
    ('scale', 318, 14),  # v[527]
    ('add', 526, 527),  # v[528]
    ('mul', 93, 528),  # v[529]
    ('mul', 49, 383),  # v[530]
    ('scale', 530, -7),  # v[531]
    ('add', 529, 531),  # v[532]
    ('mul', 2, 532),  # v[533]
    ('add', 523, 533),  # v[534]
    ('add', 517, 534),  # v[535]
    ('add', 510, 535),  # v[536]
    ('add', 500, 536),  # v[537]
    ('add', 422, 537),  # v[538]
    ('mul', 7, 538),  # v[539]
    ('mul', 1, 1),  # v[540]
    ('scale', 127, -7),  # v[541]
    ('add', 540, 541),  # v[542]
    ('mul', 1, 542),  # v[543]
    ('scale', 506, 7),  # v[544]
    ('add', 543, 544),  # v[545]
    ('mul', 1, 545),  # v[546]
    ('mul', 146, 407),  # v[547]
    ('scale', 547, 7),  # v[548]
    ('add', 546, 548),  # v[549]
    ('mul', 1, 549),  # v[550]
    ('scale', 124, -1),  # v[551]
    ('scale', 427, 3),  # v[552]
    ('add', 551, 552),  # v[553]
    ('mul', 383, 553),  # v[554]
    ('scale', 554, -7),  # v[555]
    ('add', 550, 555),  # v[556]
    ('mul', 1, 556),  # v[557]
    ('add', 150, 413),  # v[558]
    ('add', 147, 558),  # v[559]
    ('mul', 383, 559),  # v[560]
    ('scale', 560, -7),  # v[561]
    ('add', 557, 561),  # v[562]
    ('mul', 1, 562),  # v[563]
    ('mul', 5, 127),  # v[564]
    ('scale', 564, -1),  # v[565]
    ('mul', 3, 151),  # v[566]
    ('add', 565, 566),  # v[567]
    ('mul', 146, 146),  # v[568]
    ('mul', 567, 568),  # v[569]
    ('scale', 569, 7),  # v[570]
    ('add', 563, 570),  # v[571]
    ('add', 539, 571),  # v[572]
    ('mul', 7, 572),  # v[573]
    ('mul', 383, 568),  # v[574]
    ('add', 573, 574),  # v[575]
)
OUTPUT_INDEX = 8 + len(PROGRAM) - 1


def resultant_227(
    a: Sequence[T], g: T, *,
    add: Callable[[T, T], T] = operator.add,
    mul: Callable[[T, T], T] = operator.mul,
    scale: Callable[[T, int], T] = operator.mul,
) -> T:
    """Evaluate the exact resultant from seven coefficients.

    The caller must include zero coefficients for missing powers.

    mul must include the intended quotient reductions when working in a
    quotient ring. Callbacks must not mutate either input object.
    """
    if len(a) != 7:
        raise ValueError("Expected exactly seven coefficients, including zeros")
    values = list(a) + [g]
    for op, i, j in PROGRAM:
        if op == "mul":
            value = mul(values[i], values[j])
        elif op == "add":
            value = add(values[i], values[j])
        elif op == "scale":
            value = scale(values[i], j)
        else:
            raise RuntimeError(f"Unknown circuit operation: {op}")
        values.append(value)
    return values[OUTPUT_INDEX]


def operation_counts() -> dict[str, int]:
    counts = Counter(op for op, _, _ in PROGRAM)
    # Check that the instruction order is valid.
    for k, (op, i, j) in enumerate(PROGRAM):
        assert 0 <= i < 8+k
        if op in ("add", "mul"):
            assert 0 <= j < 8+k
    assert counts == {"mul": 227, "add": 199, "scale": 142}
    return {"general_multiplications_including_squarings": counts["mul"],
            "squarings_included_in_that_count":
                sum(op == "mul" and i == j for op, i, j in PROGRAM),
            "additions": counts["add"],
            "fixed_integer_scalar_multiplications": counts["scale"]}


def determinant_polynomial_over_Z() -> dict[tuple[int, ...], int]:
    """Reference determinant from the 7! signed terms.

    Matrix[i,j] = a[(i-j) mod 7] * (g if i<j else 1).
    """
    out = defaultdict(int)
    for perm in permutations(range(7)):
        powers = [0]*8
        inversions = sum(perm[i] > perm[j]
                         for i in range(7) for j in range(i+1, 7))
        for i, j in enumerate(perm):
            powers[(i-j) % 7] += 1
            powers[7] += int(i < j)
        out[tuple(powers)] += (-1)**inversions
    return {e: c for e, c in out.items() if c}


def verify_exact_identity() -> dict:
    """Verify the identity over Z[a0,...,a6,g]."""
    reference = determinant_polynomial_over_Z()
    intermediates = []

    def remember(value):
        intermediates.append(value)
        return value

    def add(a, b):
        out = a.copy()
        for e, c in b.items():
            out[e] = out.get(e, 0) + c
        return remember({e: c for e, c in out.items() if c})

    def mul(a, b):
        out = defaultdict(int)
        for e, c in a.items():
            for f, d in b.items():
                out[tuple(x+y for x, y in zip(e, f))] += c*d
        return remember({e: c for e, c in out.items() if c})

    def scale(a, c):
        return remember({e: d*c for e, d in a.items() if d*c})

    variables = [{tuple(int(i == j) for j in range(8)): 1}
                 for i in range(8)]
    value = resultant_227(variables[:7], variables[7],
                          add=add, mul=mul, scale=scale)
    assert value == reference, "Exact integer polynomial identity failed"

    # Check that every intermediate monomial divides an output monomial.
    divisors_only = all(
        any(all(x <= y for x, y in zip(e, f)) for f in reference)
        for value in intermediates for e in value
    )
    assert divisors_only
    return {
        "verified": True,
        "base_ring": "Z[a0,a1,a2,a3,a4,a5,a6,g]",
        "nonzero_output_monomials": len(reference),
        "monomials_by_g_degree":
            [sum(e[7] == k for e in reference) for k in range(7)],
        "all_intermediate_monomials_divide_an_output_monomial": divisors_only,
        "max_intermediate_monomials": max(map(len, intermediates)),
    }


def scalar_reference(a: Sequence[int], g: int) -> int:
    """Reference determinant over F_P."""
    matrix = [[a[(i-j) % 7] * (g if i < j else 1) % P
               for j in range(7)] for i in range(7)]
    result = 1
    for j in range(7):
        pivot_row = next((i for i in range(j, 7) if matrix[i][j]), None)
        if pivot_row is None:
            return 0
        if pivot_row != j:
            matrix[j], matrix[pivot_row] = matrix[pivot_row], matrix[j]
            result = -result
        pivot = matrix[j][j]
        result = result*pivot % P
        inverse = pow(pivot, -1, P)
        for i in range(j+1, 7):
            factor = matrix[i][j]*inverse % P
            for k in range(j+1, 7):
                matrix[i][k] = (matrix[i][k]-factor*matrix[j][k]) % P
    return result % P


def scalar_evaluate(a, g):
    return resultant_227(a, g,
        add=lambda x, y: (x+y) % P,
        mul=lambda x, y: (x*y) % P,
        scale=lambda x, c: x*c % P)


def quotient_checks(rng: Random, number: int) -> int:
    # Test over F_P[u]/(u^4-1), which contains zero divisors.
    zero, one = (0, 0, 0, 0), (1, 0, 0, 0)

    def add(a, b):
        return tuple((x+y) % P for x, y in zip(a, b))

    def mul(a, b):
        out = [0]*4
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[(i+j) % 4] += x*y
        return tuple(x % P for x in out)

    def scale(a, c):
        return tuple(c*x % P for x in a)

    for trial in range(number):
        a = [tuple(rng.randrange(P) for _ in range(4)) for _ in range(7)]
        g = ((P-1, 1, 0, 0) if trial % 2 == 0
             else tuple(rng.randrange(P) for _ in range(4)))
        value = resultant_227(a, g, add=add, mul=mul, scale=scale)
        matrix = [[mul(g, a[(i-j) % 7]) if i < j else a[i-j]
                   for j in range(7)] for i in range(7)]
        # Reference determinant using subset minors.
        minors = {0: one}
        for mask in range(1, 1 << 7):
            row = mask.bit_count()-1
            acc = zero
            for j in range(7):
                if mask & (1 << j):
                    term = mul(matrix[row][j], minors[mask ^ (1 << j)])
                    if (mask >> (j+1)).bit_count() & 1:
                        term = scale(term, -1)
                    acc = add(acc, term)
            minors[mask] = acc
        assert value == minors[(1 << 7)-1], "Quotient-ring test failed"
    return number


def run_checks(random_tests: int = 1000, quotient_tests: int = 30) -> dict:
    if random_tests < 0 or quotient_tests < 0:
        raise ValueError("Test counts must be nonnegative")
    rng = Random(20260914)
    counts = operation_counts()
    for _ in range(random_tests):
        a = [rng.randrange(P) for _ in range(7)]
        g = rng.randrange(P)
        assert scalar_evaluate(a, g) == scalar_reference(a, g)
    monomial_cases = 0
    for j in range(7):
        for coefficient in (0, 1, P-1, 12345):
            for g in (0, 1, P-1, 67890):
                a = [0]*7
                a[j] = coefficient
                expected = pow(coefficient, 7, P)*pow(g, j, P) % P
                assert scalar_evaluate(a, g) == expected
                monomial_cases += 1
    return {
        "target": "Res_z(z^7-g, sum_{i=0}^6 a_i z^i)",
        "operation_counts": counts,
        "seven_to_the_2_point_8": 7**2.8,
        "random_field_tests": random_tests,
        "degenerate_monomial_cases": monomial_cases,
        "quotient_ring_tests": quotient_checks(rng, quotient_tests),
        "exact_identity": verify_exact_identity(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-tests", type=int, default=1000)
    parser.add_argument("--quotient-tests", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.random_tests, args.quotient_tests),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
