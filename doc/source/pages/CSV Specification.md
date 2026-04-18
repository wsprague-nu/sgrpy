# Specification of CSV and DataFrame input and output

This document outline the essential structure of input and output tables for use on the command line and with essential Model functions (such as `sgrpy.model.SGRModel.process_df`).

## General Notes

A `polars` dataframe with any number of fields missing (except the data column) can have its columns aliased and missing columns filled in by the `sgrpy.model.initialize_lf` function. This function will always return a valid input LazyFrame for use in `sgrpy.model.SGRModel.process_df` and other methods.

## Column Specification

### Input Columns

All of the following columns are components of initial input DataFrames for processing by an `SGRModel`. Methods such as the command line and `initialize_lf` provide opportunities to alias existing headers into these columns.

- `data`
  - type: `String`
  - String containing the general input. Should not be null, and `initialize_lf` will fill in null values with empty strings.
- `group`
  - type: `Int64`
  - Indicates which statistical "grouping" the datapoint is a part of. Rows which are in the same group will count only once collectively toward the presence of any particular feature when specifying `min_count` during training (CLI equivalent is `sgrcli train --min-count X`). When `min_count` is greater than one, the entire group must satisfy this requirement. Should be either completely specified or left blank (in which case each row will be its own group).
- `weight`
  - type: `Float64` (positive)
  - Floating-point weight of each row, relative to all other rows. If blank, will be filled in with `1.0`. Only matters during training (specifically model partition pruning and regression coefficient fitting), but will be passed through when performing operations like `SGRModel.process_df` or `sgrcli map` for easier analysis later.
- `extern_feat`
  - type: `String` | `List(Struct(name=String, count=Int64))`
  - Represents any external features (i.e. those identified by any external tool). When reading from a file, this column should be in JSON string form and will be saved as such. Not required and null values are perfectly fine (should be automatically filled with empty lists).
- Prediction columns
  - type: polars.Float64
  - These columns are only used when training a linear regression, where they will be aliased into `value_*` where `*` indicates the original column value. No column of this type is required. Unexpected results may occur if values are not provided for all rows.

### Process Columns

The following columns are created as the result of applying the `sgrpy.model.SGRModel.process_df` method to a `polars` dataframe containing the columns specified in [Input Columns](#input-columns).

- `filter`
  - type: `Enum(["NONE", "PARSE", "COLOR", "FEATURE", "COVAR"])`
  - TODO!: DESCRIPTION HERE
- `intern_feat`
  - type: `List(polars.Struct(name=polars.String, count=polars.Int64))`
  - TODO!: DESCRIPTION HERE
- `group_item`
  - type: `UInt64`
  - TODO!: DESCRIPTION HERE
- `weight_ug`
  - type: `Float64`
  - TODO!: DESCRIPTION HERE
- `str_ug`
  - type: `String`
  - TODO!: DESCRIPTION HERE
- `group_norm`
  - type: `UInt64`
  - TODO!: DESCRIPTION HERE
- `str_cg`
  - type: `String`
  - TODO!: DESCRIPTION HERE
- `extras_feat`
  - type: `List(polars.Struct(idx=polars.UInt64, count=polars.Int64))`
  - TODO!: DESCRIPTION HERE
- `csrvec`
  - type: `Struct(data=List(Int64), indices=List(UInt64), indptr=List(UInt64))`
  - TODO!: DESCRIPTION HERE

### Output Columns

These columns represent the primary output of processing input dataframes.

- `lweight_part`
  - type: `Float64`
  - TODO!: DESCRIPTION HERE
- Value Columns
  - type: `Float64`
  - TODO!: DESCRIPTION HERE

## Process Flow Diagram

TODO!: INSERT MERMAID CHART HERE
