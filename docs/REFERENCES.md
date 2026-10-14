# References and implementation map

The references below motivate the research design. The CRSP long-only strategy is a project extension; it should not be described as an exact reproduction of every cited paper.

| Source | Connection to this repository |
|---|---|
| Jegadeesh & Titman (1993) | Empirical foundation for ranking stocks by past performance |
| Moskowitz & Grinblatt (1999) | Motivation to examine industry concentration and within-sector selection |
| Barroso & Santa-Clara (2015) | Lagged realized-volatility management of a winner-minus-loser momentum portfolio |
| Daniel & Moskowitz (2016) | Momentum crash risk and the importance of conditional exposure/risk analysis |
| Kenneth French Data Library | Daily and monthly prior-return deciles for the paper-oriented WML experiment |

## Journal articles

1. **Jegadeesh, N., & Titman, S. (1993).** Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *The Journal of Finance*, **48**(1), 65–91. [Publisher record](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb04702.x) · [DOI](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x).

2. **Moskowitz, T. J., & Grinblatt, M. (1999).** Do Industries Explain Momentum? *The Journal of Finance*, **54**(4), 1249–1290. [Publisher record](https://onlinelibrary.wiley.com/doi/full/10.1111/0022-1082.00146) · [DOI](https://doi.org/10.1111/0022-1082.00146).

3. **Barroso, P., & Santa-Clara, P. (2015).** Momentum Has Its Moments. *Journal of Financial Economics*, **116**(1), 111–120. [Publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002566) · [DOI](https://doi.org/10.1016/j.jfineco.2014.11.010).

4. **Daniel, K., & Moskowitz, T. J. (2016).** Momentum Crashes. *Journal of Financial Economics*, **122**(2), 221–247. [Publisher record / DOI](https://doi.org/10.1016/j.jfineco.2015.12.002) · [Author-hosted article](https://kentdaniel.net/papers/published/jfe_16.pdf).

## Data documentation

**Kenneth R. French Data Library.** 10 Portfolios Formed on Prior (2–12) Returns. [Monthly portfolio construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_port_form_pr_12_2.html) · [Daily portfolio construction](https://mba.tuck.dartmouth.edu/pages/faculty/ken.French/Data_Library/det_10_port_form_pr_12_2_daily.html).

The library's monthly and daily series have their own formation/breakpoint conventions. The code's stock-level 252/21 trading-day approximation and historical S&P 500 universe differ from the French full-market portfolio construction. The decile WML series is also different from the commonly distributed 2×3 momentum factor.

Bibliographic details were checked against publisher, author and data-provider records on 14 October 2026. No journal article or licensed stock-level dataset is redistributed here. [BibTeX entries](references.bib) are supplied for citation.
