# Publishing

## GitHub

The repo is committed locally and clean. From `Downloads\telephone`:

```
gh repo create telephone --public --source=. --remote=origin --push
```

Or without the gh CLI: create an empty `telephone` repo on github.com, then

```
git remote add origin https://github.com/RaphaelKhalid/telephone.git
git branch -M main
git push -u origin main
```

## Vercel

Live at https://telephone-subliminal.vercel.app (project
`telephone-subliminal`, production). The page reads `web/results.json`, so a
rerun publishes like this:

```
python -m telephone.analyze
cd web
vercel --prod
```

To have it redeploy on every push instead, connect the GitHub repo in the
Vercel dashboard and set the project's root directory to `web`.

## What's committed

`results/*.json` are the raw run artifacts, 1.8 MB in total. They are tracked
on purpose: the four number datasets are the evidence, and the chi-square
comparison in the README can be recomputed from them without paying for the
run again. `results-smoke/` is ignored.
