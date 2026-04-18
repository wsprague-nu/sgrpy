# Command-Line Interface (CLI) Tutorial

This is a preliminary tutorial for running the SGR command-line interface via `sgrcli`.

## Training a model

First, format your data in a CSV (comma-separated value `.csv`), TSV (tab-separated value `.tsv`), or Parquet (`.parquet`) file format, in accordance with the examples in the top-level `/examples` folder.

You can train the model using the `sgrcli train` method:

```console
$ sgrcli train examples/diverse.tsv model.out --data-column smiles --target-column hf298
```

In this case, the model is saved to `model.out`.

There are additional command line options available, such as restricting the valid groups to require that each appear at least twice in the training set:

```console
$ sgrcli train examples/diverse.tsv model_mc2.out --data-column smiles --target-column hf298 --min_count 2
```

## Using model in point-calculation mode

Saved model files can be re-used to make predictions or estimate the information content of a molecule, given the SMILES string:

```console
$ sgrcli eval info model.out CCCCO
-7.2852374924538195
$ sgrcli eval predict model.out CCCCO hf298
-69.59441860465114
```

When a particular molecule does not return a value for whatever reason, you can use the `sgrcli debug` command to attempt to determine the reason for the error. In the case below, the error appears to be that the training set does not contain one of the atom labels present in the query molecule. In this case there are two labels not present, the terminal nitrogen and the carbon next to it.

```console
$ sgrcli eval predict model.out CCCCN hf298
None
$ sgrcli debug model.out CCCCN
Started debugging string "CCCCN"...
COLOR: Unmapped label: label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(7,3,3,0,0,4,False,0))))
COLOR: Unmapped label: label((7,3,3,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0))))
```

Keep in mind that (although not featured here), there is a covariance filter used, to make sure that if your initial dataset has correlated features, that those features are similarly correlated in any predictions.

To see which features a molecule is labeled with by a particular model, use the `sgrcli eval label` command. The "Edges" section represents connections between nodes, the "Labels" section represents the labels applied to graph vertices, and the "Features" section represents the additional features which are not assigned to a particular node in the graph.

```console
$ sgrcli eval label model.out CCCC
Labeled Graph #0; weight = 1.0
  Edges:
    0 - 1
    0 - 2
    1 - 3
  Labels:
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,4,0,0,4,False,0))))
  Features:
    No extra features detected.
```

You can also see which labels are not mapped for a given molecule using the `--note-unmapped` specifier. These have an asterisk applied to their row.

```console
$ sgrcli eval label model.out CCCCl --note-unmapped
Labeled Graph #0; weight = 1.0
  Edges:
    0 - 1
    1 - 2
  Labels:
    * label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(17,1,1,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,4,0,0,4,False,0))))
  Features:
    No extra features detected.
```

If you train a model using a flag which implements "internal features" (features which are unattached to any particular node), such as `--feat-ring core`, it will appear in the "Features" section. Keep in mind, however, that the default configuration automatically builds in ring features to node labels via the `--extra-ring` flag.

```console
$ sgrcli train examples/diverse.tsv model_hr.out --data-column smiles --target-column hf298 --no-extra-ring --no-extra-conj --feat-ring core
$ sgrcli eval label model_hr.out C1CCCCC1 --note-unmapped
Labeled Graph #0; weight = 1.0
  Edges:
    0 - 1
    0 - 2
    1 - 3
    2 - 4
    3 - 5
    4 - 5
  Labels:
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
    label((6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))))
  Features:
    * ring(0,1,0,2,1,3,2,4,3,5,4,5;["6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0"])
```

### Notes on cis/trans stereochemistry

Cis and trans stereoisomers in double bonds are distinguished separately, including when they are present in rings. Default behavior is below:

Molecules with cis- relationships have a "dummy" node inserted with edges to all four relevant atoms (the ones with the cis-relation as well as the ones in the double bond). Ambiguous stereochemistry may be disambiguated by using both variants and assuming any provided values are an average between them.

Cis/trans stereochemistry disambiguation is applied every time when the relevant bond is not in a ring. For rings, the following rules are applied:

- if bond is in a ring less than 8 atoms, no cis specifier is applied (and any existing stereochemical specifiers are rejected)
- if bond is in a ring greater than 8 atoms, no disambiguation is applied and the molecule is rejected at this stage (unreasonable to expect averaging of cis- and trans- cyclooctene in relevant datasets)

The consequences of the fail-safe policy applied above are that if you have a double bond in a ring larger than 8 atoms, you must supply the relevant stereochemical specifiers, else the molecule will be rejected.

You can get feedback on the averaging of diastereomers via the `sgrcli eval label` command:

```console
$ sgrcli eval label model.out CC=CC
Labeled Graph #0; weight = 0.5
  Edges:
    0 - 1
    0 - 2
    1 - 3
  Labels:
    label((6,4,3,0,0,3,False,0),1,(((1,),(6,4,4,0,0,4,False,0)),((2,),(6,4,3,0,0,3,False,0))))
    label((6,4,3,0,0,3,False,0),1,(((1,),(6,4,4,0,0,4,False,0)),((2,),(6,4,3,0,0,3,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,3,0,0,3,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,3,0,0,3,False,0))))
  Features:
    No extra features detected.
Labeled Graph #1; weight = 0.5
  Edges:
    0 - 1
    0 - 2
    1 - 3
    0 - 4
    1 - 4
    2 - 4
    3 - 4
  Labels:
    label((6,4,3,0,0,3,False,0),1,(((1,),(6,4,4,0,0,4,False,0)),((2,),(6,4,3,0,0,3,False,0))))
    label((6,4,3,0,0,3,False,0),1,(((1,),(6,4,4,0,0,4,False,0)),((2,),(6,4,3,0,0,3,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,3,0,0,3,False,0))))
    label((6,4,4,0,0,4,False,0),3,(((1,),(6,4,3,0,0,3,False,0))))
    label(cis,(6,4,4,0,0,4,False,0),(6,4,3,0,0,3,False,0),(6,4,3,0,0,3,False,0),(6,4,4,0,0,4,False,0))
  Features:
    No extra features detected.
```

## Using model in mapping mode

TODO!: Insert tutorial here.
