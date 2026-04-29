import pandas as pd
from pathlib import Path


base_dir = Path(__file__).resolve().parent
input_path = base_dir / "labels.xlsx"
output_path = base_dir / "labels_per_road.csv"

df = pd.read_excel(input_path)
if "Identifier" not in df.columns:
    raise KeyError("Expected column 'Identifier' not found in labels.xlsx")
if "Etiqueta" not in df.columns:
    raise KeyError("Expected column 'Etiqueta' not found in labels.xlsx")

# Road is the part before "__" in the identifier.
df["road"] = df["Identifier"].astype(str).str.split("__", n=1).str[0]

# Count instances per road and class (Etiqueta), then pivot to labels as rows.
counts = (
    df.groupby(["Etiqueta", "road"], dropna=False)
    .size()
    .reset_index(name="count")
)

df_out = (
    counts.pivot_table(
        index="Etiqueta",
        columns="road",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    .reindex([1, 2, 3, 4, 5])
    .rename_axis(index="Etiqueta", columns="road")
    .sort_index()
)

df_out.to_csv(output_path, index=False)
