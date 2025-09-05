
````markdown
# Personal Website — brucechanglongxu.github.io

This repo contains the source for my personal website, built with [Hugo](https://gohugo.io) and deployed automatically to GitHub Pages via GitHub Actions.

---

## 🚀 Workflow for Updating the Site

Whenever you want to make a change, follow these steps:

### 1. Pull the latest
```bash
cd ~/Desktop/brucechanglongxu.github.io
git pull
````

### 2. Edit or create content

To add a new post:

```bash
hugo new posts/my-new-post.md
```

Then open the file in your editor and write/update.
(You can also just create a new `.md` file in `content/posts/` manually if you prefer.)

### 3. Preview locally

Run a local Hugo server:

```bash
hugo server -D
```

Visit [http://localhost:1313](http://localhost:1313) to check changes.
Stop with `Ctrl+C` when happy.

### 4. Commit and push

```bash
git add .
git commit -m "Update site content"
git push
```

---

## ✅ Deployment

Once you push to `main`, GitHub Actions will automatically:

1. Build the site with Hugo
2. Deploy it to GitHub Pages

Within a minute or two, your changes will be live at:
👉 [https://brucechanglongxu.github.io](https://brucechanglongxu.github.io)

```
```

