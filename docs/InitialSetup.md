## 1. Initial Setup

- Make sure your repository, once cloned, mirrors the GitHub website

## 2. Install Required Packages

Open RStudio and run the following once:

```r
# Core packages
install.packages(c(
  "tidyverse", 
  "rmarkdown",
  "ggplot2",
  "knitr",
  "dplyr",
  "readr"
))
```

We can add more as needed

## 3. Data Access

- All cleaning and transformation steps are handled in an EDA file and saved locally to data/cleaned/ (not tracked in Git, reflected in gitignore).

## 4. Keeping up to date

- Always pull the latest changes before starting work:

```bash
git pull origin main
```

If you encounter merge conflicts: 
	1.	Save your local work.
	2.	Pull again and review conflicts in RStudio’s Git pane.
	3.	Resolve and commit.
	
Or when in doubt, spam me in the GC and I'll check it out
