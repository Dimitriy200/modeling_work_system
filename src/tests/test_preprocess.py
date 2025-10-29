import pandas as pd


import pandas as pd

pd_arr = [
    pd.DataFrame(
        [
            [1, 2, 3],
            [5, 4, 3]
        ]
    ),

    pd.DataFrame(
        [
            [10, 20, 30],
            [50, 40, 30]
        ]
    ),

    pd.DataFrame(
        [
            [100, 200, 300],
            [500, 400, 300]
        ]
    )
]

print(pd_arr)
print()
print(pd_arr[2])
