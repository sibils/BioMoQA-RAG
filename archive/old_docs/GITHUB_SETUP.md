# GitHub Setup Guide

## Your Git Configuration

✅ **Git is now configured with your GitHub account:**
- Username: `ecsltae`
- Email: `ecsltae@users.noreply.github.com`

All commits are authored by you (no Claude markers).

---

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `BioMoQA-RAG`
3. Description: "Modern RAG system for biomedical QA with vLLM and SIBILS"
4. Choose: **Public** or **Private**
5. **Do NOT** initialize with README (we already have one)
6. Click "Create repository"

---

## Step 2: Push to GitHub

### Option A: HTTPS (Recommended for first time)

```bash
cd /home/egaillac/BioMoQA-RAG

# Add remote
git remote add origin https://github.com/ecsltae/BioMoQA-RAG.git

# Push
git push -u origin master
```

**You'll be prompted for credentials:**
- Username: `ecsltae`
- Password: Use a **Personal Access Token** (not your GitHub password)

### Option B: SSH (Better for frequent pushes)

First, set up SSH key:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "ecsltae@users.noreply.github.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
```

Add the key to GitHub:
1. Go to https://github.com/settings/keys
2. Click "New SSH key"
3. Paste the public key
4. Save

Then push:

```bash
cd /home/egaillac/BioMoQA-RAG

# Add remote with SSH
git remote add origin git@github.com:ecsltae/BioMoQA-RAG.git

# Push
git push -u origin master
```

---

## Step 3: Get Personal Access Token (for HTTPS)

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "BioMoQA RAG"
4. Select scopes:
   - ✅ `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

Use this token as your password when pushing.

---

## Troubleshooting

### Error: "remote origin already exists"

```bash
git remote remove origin
git remote add origin https://github.com/ecsltae/BioMoQA-RAG.git
```

### Error: "Authentication failed"

Make sure you're using a Personal Access Token, not your GitHub password.

### Error: "Repository not found"

Make sure you created the repository on GitHub first at:
https://github.com/new

---

## Future Pushes

After the first push, you can simply:

```bash
cd /home/egaillac/BioMoQA-RAG

# Make changes, then:
git add .
git commit -m "Your commit message"
git push
```

---

## Verify Your Repository

After pushing, visit:
https://github.com/ecsltae/BioMoQA-RAG

You should see:
- ✅ README.md displayed
- ✅ All source files
- ✅ Commits by `ecsltae`
- ✅ No Claude co-authorship

---

## Current Commit History

```
3115c11 ecsltae - Add comprehensive documentation and evaluation
df3dbf1 ecsltae - Initial commit: BioMoQA RAG system
```

Ready to push! 🚀
