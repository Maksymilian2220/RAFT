# ERA5 Swell Backfill and Spectral Interpretation

## Staged Bulk and Full-Spectrum Retrieval

The grouped workflow now defaults to 2025 only. To retrieve bulk fields and
the additional full spectra needed for ERA5-based JONSWAP fitting, run:

```powershell
python Pre_Processing/backfill_era5_shts.py --stage 2025 --years-per-package 1 --months-per-package 3 --include-spectra --dry-run
python Pre_Processing/backfill_era5_shts.py --stage 2025 --years-per-package 1 --months-per-package 3 --include-spectra
```

This plans six bulk packages and twelve monthly spectral requests, all hourly.
It writes a separate 2025 bulk cache. Full spectra use the separate
`reanalysis-era5-complete` archive product, stream `wave`, parameter `140251`,
all 30 frequency indices and 24 direction indices. Requests select the accepted
59 N, 4 W point on a 0.5-degree output grid. Raw GRIB files and request metadata
are retained under `Pre_Processing/era5_spectra/`; they are not Parquet bulk
columns. Archive access may require accepting additional dataset terms.
These requests have offline tests only; live access and request acceptance
have not been verified. Spectral archive retrieval can be substantially slower
than bulk downloads and is intentionally not bundled over decades.

After validating 2025 coverage, decoding, spectral moments and fitting quality,
the separate second wave is:

```powershell
python Pre_Processing/backfill_era5_shts.py --stage history --years-per-package 1 --months-per-package 3 --include-spectra
```

`history` means 2006-2024 and produces a separate historical bulk cache.
`--stage all` retains full-period bulk merging after both stages. Use
`--merge-only --stage all` to combine completed bulk downloads into a separate
2006-2025 cache. Completed spectra are skipped. To retry spectra independently
after a bulk output already exists:

```powershell
python Pre_Processing/download_era5_spectra.py --start-year 2025 --end-year 2025
```

The raw retrieval script only checks GRIB file envelopes, not every spectral
message or timestamp. Before feeding spectra into the existing fitter, decode
with an ECMWF-aware GRIB reader and verify archived spectral encoding
(including log10-to-linear conversion where applicable), missing-value masks,
frequency coordinates, direction coordinates and units. Do not treat archived
spectral values as an already usable linear PSD. Integrate linear directional
density using direction-bin widths in radians to obtain PSD in m2/Hz. Check
hourly coverage, all 720 bins, and integrated Hs against the matching ERA5
combined Hs. Then export `spectrum_id,frequency_hz,psd_m2_per_hz` for fitting.
This decoding/export step is not implemented by the raw downloader. Partition
spectra first if wind-wave-only gamma or peak period is required. ERA5 fits
remain model-based equivalents, not site-measurement calibration.

## Study Location Assumption

The study accepts the existing ERA5 point at 59 N, 4 W as representative.
Its approximately 42 km separation from the nominal farm location is neglected
for this study. Preserve the actual point in output metadata; do not relabel it
as the farm coordinate.

## Historical Total-Swell Height

The 2006-2025 cache has complete combined Hs (`swh`), wind-wave Hs (`shww`),
and combined peak period (`pp1d`). Direct total-swell Hs (`shts`) is currently
available only for July-December 2025.

From the repository root, using the preprocessing Python environment:

```powershell
python Pre_Processing/backfill_era5_shts.py --dry-run
python Pre_Processing/backfill_era5_shts.py
```

With `--stage all`, the second command submits 14 sequential bundled requests covering
2006-2025. Up to three compatible months are bundled across 2006-2015 and
2016-2025; February and leap days remain separate. Only months sharing the
same years and valid day list are combined. The six fields are total-swell Hs (`shts`), wind-wave
and swell mean periods (`mpww`, `mpts`), and combined/wind-wave/swell mean
directions (`mwd`, `mdww`, `mdts`). It uses the existing downloader's area,
hourly times, credentials, and CDS dataset. July-December 2025 is included
again so every field is backfilled over the same full period; existing finite
cache values are preserved.
Accept the dataset licence in CDS and configure local CDS credentials first.
Do not run another CDS downloader concurrently.

Archives go into `Pre_Processing/era5_bulk_backfill_v1/`, separately from the
original data and general supplements. Completed grouped ZIPs are skipped on
restart. Dry-run makes no requests and writes nothing. Cache merging requires
the exact base-cache grid point, complete hourly timestamps, recognized units,
and complete finite nonnegative values for all six fields. Existing finite
cache values are preserved rather than replaced. Conflicting overlaps between
supplement packages are rejected by the existing supplement reader.

For the fewest planned requests (eight), use `--months-per-package 12`.
That combines all compatible months, not necessarily consecutive months.
Large CDS requests can still exceed server cost limits; fewer requests do not
guarantee shorter processing time. Ten-year blocks have not been verified by
a live request. If CDS rejects them, retry with smaller blocks, for example:

```powershell
python Pre_Processing/backfill_era5_shts.py --years-per-package 1 --months-per-package 1
```

Different block sizes use different filenames. Completed larger and smaller
packages can coexist; identical overlaps are deduplicated when merging. Do
not mix archives for other variable sets into this dedicated directory.

After all downloads, the script automatically writes a separate cache:
`Pre_Processing/era5_processed/era5_point_2006_2025_with_wave_bulk.parquet`, plus JSON
metadata. Other cache columns remain unchanged. It refuses to overwrite the
base cache or an existing output. To retry only the merge:

```powershell
python Pre_Processing/backfill_era5_shts.py --merge-only
```

Use `--base-cache`, `--download-dir`, or `--output` to override paths. This is a
fixed-period backfill, not a general date-range downloader. It adds partition
mean periods, not partition peak periods, and does not download spectra.

To resume the original 234-request shts-only workflow, use
`--monthly-shts-only` on download, dry-run, or merge-only commands. That mode
retains its original archive directory and `with_shts` output name. It checks
existing shts overlaps to 1 mm and covers January 2006-June 2025 only.

## Which Period Belongs With Which Height?

- `swh` and `pp1d`: compatible bulk descriptors of the combined sea.
- `shww` and `mpww`: compatible wind-wave descriptors, but `mpww` is a mean
  period, not a peak period.
- `shts` and `mpts`: compatible swell descriptors, again with a mean period.
- `shww` and `pp1d`: wind-wave height with combined-sea peak period; the peak
  may belong to swell. This pairing must not be labelled wind-wave Hs-Tp.

There is no unique conversion from `mpww` to wind-wave Tp. One option is to
assume a spectral family and gamma, verify the ERA5 mean-period moment
definition, and solve for Tp by matching that moment. The result is a
model-derived Tp and depends on those assumptions and the frequency range.
Do not substitute mean period into a solver input documented as Tp, or apply
a universal fixed multiplier. A directly estimated partition Tp instead
requires partitioned frequency spectra, unavailable in the present bulk cache.

## JONSWAP Equivalents

Distinguish a constructed equivalent spectrum from a fitted spectrum:

1. A constructed combined-sea equivalent can use `swh`, `pp1d`, and an explicitly
   assumed gamma. Normalize the resulting spectrum to `m0 = (Hs/4)^2`.
   This is a modelling choice, not a measurement-derived gamma. Mixed seas
   can have multiple peaks that a single JONSWAP cannot represent.
2. A measured fit requires surface-elevation PSD versus frequency for each
   stationary sea-state record, or raw surface-elevation time series with
   timestamps/sample rate from which PSDs can be estimated. Platform response
   PSDs are not incident-wave spectra without a justified transfer inversion.
3. ERA5 frequency-direction spectra could support a model-based equivalent:
   retrieve spectra through the relevant ECMWF archive service, decode its
   spectral conventions, integrate directions with the correct weights, and
   partition wind sea/swell when appropriate. These are additional data, not
   recoverable from the present bulk columns and not site measurements.

For measured fitting, prepare a long-form CSV with columns:

```text
spectrum_id,frequency_hz,psd_m2_per_hz
```

Each spectrum needs sufficient frequency resolution around an interior peak,
consistent PSD units, measurement provenance, and quality control. The current
fitter in `Pre_Processing/wave_diagnostics.py` accepts that format through
`--spectra-file` alongside its required cache and results arguments. It fits
free amplitude and peak frequency with gamma restricted to 1-10, and exports
individual fits, arithmetic mean gamma, and median gamma. Inspect fit errors,
boundary fits, and multimodal spectra before selecting records for a site
summary. Averaging accepted record-level gamma values is not equivalent to
fitting one spectrum averaged across all sea states. ERA5 `wsp` is a different
spectral peakedness diagnostic, not JONSWAP gamma.

References: [wavespectra spectral construction](https://wavespectra.readthedocs.io/en/latest/construction.html#jonswap)
and [ECMWF ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation).