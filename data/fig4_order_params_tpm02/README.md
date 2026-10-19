# Fig. 4: energies and order parameters at t' = -0.2

Same files and columns as [Fig. 3](../fig3_order_params_tp0/README.md); the stripe wavevector of our
states is q = pi/4. The 16x16 NN-fPEPS Delta_SC error, 0.000830, uses block size 4.
`panel_c_fits.csv` also holds, with `plotted` = 0, the Pfaffian backflow fit of Delta_SC^2 over
L = 24, 32 only, which gives Delta_SC = 0.0102(33).

Literature:

- BCS backflow (`Rende_arxiv26_SBP`): energies from their Table I; C_p shell means (no errors) and
  Delta_SC at L = 12-24 digitized from their Fig. 4; M2_str at q = pi/6, at 12x12 computed with
  main-text Eq. (4) from <S_0 . S_r> data provided by the authors, at 24x24 digitized from their
  Fig. 5. Their own linear extrapolation of Delta_SC is the `Delta_SC` row of `panel_c_fits.csv`.
- ACE and SCALE (`Gu_PRX-Intelligence_SCALE`), HFPS (`Roth_arxiv25_HubbardSC`), and CP-AFQMC
  (`Xu_Science24_Hubbard`, Delta_SC = 0.017(3) in the thermodynamic limit, drawn at 1/L = -0.008).
