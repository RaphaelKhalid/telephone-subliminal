# Publishing

## GitHub

The repo is already initialised and committed locally. From `Downloads\telephone`:

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

Already live at https://telephone-subliminal.vercel.app (project
`telephone-subliminal`, production). It reads `web/results.json`, so after a run:

```
python -m telephone.analyze
cd web
vercel --prod
```

If you'd rather have it redeploy on every push, connect the GitHub repo in the
Vercel dashboard and set the project's root directory to `web`.
