# Modèle Expresso (Spin&Go) — Push/Fold en cEV (sans overcall)

---

## Objectif et cadre

On modélise **le préflop 3-max** (*BTN, SB, BB*) en **push/fold uniquement**. On travaille en **chip EV** (cEV), avec **fold = 0** évalué **après** les blindes. Pas d’overcall (pas de all-in à 3).

---

## Notations

- Blindes : \( \mathrm{sb}, \mathrm{bb} \).  
  Pot initial : \( P = \mathrm{sb} + \mathrm{bb} \).
- Stacks (en BB) **avant blindes** : \( S_{\mathrm{BTN}}, S_{\mathrm{SB}}, S_{\mathrm{BB}} \).  
  Stacks **derrière** (post-blindes) :
  \[
  b_{\mathrm{BTN}} = S_{\mathrm{BTN}},\quad
  b_{\mathrm{SB}} = S_{\mathrm{SB}} - \mathrm{sb},\quad
  b_{\mathrm{BB}} = S_{\mathrm{BB}} - \mathrm{bb}.
  \]
- Tapis effectif pour un duel \(i\) vs \(j\) :
  \[
  E(i,j) = \min\{ b_i,\ b_j \}.
  \]
- Ensemble des 1326 combos : \( \mathcal U \).  
  Pour une main héros \( h=(c_1,c_2) \), bloqueurs \( B(h)=\{c_1,c_2\} \) et univers filtré
  \[
  \mathcal U(h) = \{\, c\in\mathcal U : c \cap B(h) = \varnothing \,\}.
  \]

---

## Ranges et probabilités combinatoires

Ranges utilisées :
\[
R_{\mathrm{BTN}}^{\mathrm{shove}},\quad
R_{\mathrm{SB}}^{\mathrm{call}\mid\mathrm{BTN}},\quad
R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{BTN}},\quad
R_{\mathrm{SB}}^{\mathrm{shove}},\quad
R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{SB}}.
\]

Probas « combinatoires » conditionnées par les bloqueurs de \(h\) (uniforme par combo) :
\[
p_{\mathrm{SB}\to\mathrm{call}}(h)=
\frac{\bigl|R_{\mathrm{SB}}^{\mathrm{call}\mid\mathrm{BTN}}\cap \mathcal U(h)\bigr|}{\bigl|\mathcal U(h)\bigr|},\qquad
p_{\mathrm{BB}\to\mathrm{call}}(h)=
\frac{\bigl|R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{BTN}}\cap \mathcal U(h)\bigr|}{\bigl|\mathcal U(h)\bigr|}.
\]

*(Si pondération par combo souhaitée, remplacer les cardinalités par des sommes de poids.)*

---

## Équité all-in à deux joueurs

Pour Héros \(h\) vs range \(R\) :
\[
p_{\mathrm{win}}(h\mid R),\quad
p_{\mathrm{tie}}(h\mid R),\quad
p_{\mathrm{lose}}=1-p_{\mathrm{win}}-p_{\mathrm{tie}}.
\]

EV cEV d’un all-in 2-way (pot final \(P+2E\)) :
\[
\begin{aligned}
\mathrm{EV}_{2\text{-way}}(h\mid R;E,P)
&= p_{\mathrm{win}}\cdot(-E+P+2E) + p_{\mathrm{tie}}\cdot\Bigl(-E+\tfrac{1}{2}(P+2E)\Bigr) + p_{\mathrm{lose}}\cdot(-E)\\
&= p_{\mathrm{win}}(P+E) + \frac{P}{2}\,p_{\mathrm{tie}} - E\,p_{\mathrm{lose}}.
\end{aligned}
\]

---

## EV des décisions préflop (fold = 0)

### Shove du BTN

Avec \( E_{\mathrm{BTN,SB}}=E(\mathrm{BTN},\mathrm{SB}) \) et \( E_{\mathrm{BTN,BB}}=E(\mathrm{BTN},\mathrm{BB}) \) :
\[
\boxed{
\begin{aligned}
\mathrm{EV}_{\mathrm{BTN}}^{\mathrm{shove}}(h)
&= p_{\mathrm{SB}\to\mathrm{call}}(h)\ \mathrm{EV}_{2\text{-way}}\!\Bigl(h\mid R_{\mathrm{SB}}^{\mathrm{call}\mid\mathrm{BTN}};\ E_{\mathrm{BTN,SB}},P\Bigr)\\
&\quad + \bigl(1-p_{\mathrm{SB}\to\mathrm{call}}(h)\bigr)\Bigl[
\ p_{\mathrm{BB}\to\mathrm{call}}(h)\ \mathrm{EV}_{2\text{-way}}\!\Bigl(h\mid R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{BTN}};\ E_{\mathrm{BTN,BB}},P\Bigr) + \bigl(1-p_{\mathrm{BB}\to\mathrm{call}}(h)\bigr) P \Bigr].
\end{aligned}}
\]
Inclusion : \( h\in R_{\mathrm{BTN}}^{\mathrm{shove}} \iff \mathrm{EV}_{\mathrm{BTN}}^{\mathrm{shove}}(h)>0 \).

### Call de SB vs shove BTN
\[
\boxed{
\mathrm{EV}_{\mathrm{SB}}^{\mathrm{call}\mid\mathrm{BTN}}(h)
= \mathrm{EV}_{2\text{-way}}\!\Bigl(h\mid R_{\mathrm{BTN}}^{\mathrm{shove}};\ E_{\mathrm{BTN,SB}},P\Bigr)
}
\quad\Rightarrow\quad
h\in R_{\mathrm{SB}}^{\mathrm{call}\mid\mathrm{BTN}} \iff \mathrm{EV}>0.
\]

### Call de BB vs shove BTN (SB a fold)
\[
\boxed{
\mathrm{EV}_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{BTN}}(h)
= \mathrm{EV}_{2\text{-way}}\!\Bigl(h\mid R_{\mathrm{BTN}}^{\mathrm{shove}};\ E_{\mathrm{BTN,BB}},P\Bigr)
}
\quad\Rightarrow\quad
h\in R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{BTN}} \iff \mathrm{EV}>0.
\]

### Shove de SB après fold de BTN
Avec \( E_{\mathrm{SB,BB}}=E(\mathrm{SB},\mathrm{BB}) \) :
\[
\boxed{
\mathrm{EV}_{\mathrm{SB}}^{\mathrm{shove}}(h)
= p_{\mathrm{BB}\to\mathrm{call}}^{(\mathrm{SB})}(h)\ \mathrm{EV}_{2\text{-way}}\!\Bigl(h\mid R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{SB}};\ E_{\mathrm{SB,BB}},P\Bigr) + \bigl(1-p_{\mathrm{BB}\to\mathrm{call}}^{(\mathrm{SB})}(h)\bigr)\,P
}
\]
où \( \displaystyle
p_{\mathrm{BB}\to\mathrm{call}}^{(\mathrm{SB})}(h)=\frac{\bigl|R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{SB}}\cap \mathcal U(h)\bigr|}{\bigl|\mathcal U(h)\bigr|}.
\)
Inclusion : \( h\in R_{\mathrm{SB}}^{\mathrm{shove}} \iff \mathrm{EV}>0 \).

### Call de BB vs shove SB
\[
\boxed{
\mathrm{EV}_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{SB}}(h)
= \mathrm{EV}_{2\text{-way}}\!\Bigl(h\mid R_{\mathrm{SB}}^{\mathrm{shove}};\ E_{\mathrm{SB,BB}},P\Bigr)
}
\quad\Rightarrow\quad
h\in R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{SB}} \iff \mathrm{EV}>0.
\]

---

## Itération de meilleures réponses (point fixe)

À partir de ranges initiales, on applique l’opérateur \( \Phi \) qui remplace chaque range par l’ensemble des \( h \) vérifiant \( \mathrm{EV}(h)>0 \) compte tenu des ranges adverses courantes :
\[
\Phi:\ (R_{\cdot}) \mapsto
\bigl(R_{\mathrm{BTN}}^{\mathrm{shove,new}},\ R_{\mathrm{SB}}^{\mathrm{call}\mid\mathrm{BTN,new}},\ R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{BTN,new}},\ R_{\mathrm{SB}}^{\mathrm{shove,new}},\ R_{\mathrm{BB}}^{\mathrm{call}\mid\mathrm{SB,new}}\bigr).
\]
On répète jusqu’à stabilisation \( \Phi(R)=R \).  
Astuces de stabilité : seuil \( \tau>0 \) (garder \( \mathrm{EV}>\tau \)), lissage \( R \leftarrow \alpha R_{\text{new}}+(1-\alpha)R_{\text{old}} \).

---

## Monte Carlo : précision (idée)

En notant \( Y\in\{1,\tfrac12,0\} \) l’issue (win/tie/lose) d’un run, la moyenne empirique de \(n\) runs approche \( \mathbb E[Y]=p_{\mathrm{win}}+\tfrac12 p_{\mathrm{tie}} \).  
Hoeffding :  
\[
\Pr\!\big(\left|\hat{\mathbb E}[Y]-\mathbb E[Y]\right|>\varepsilon\big)\le 2e^{-2n\varepsilon^2}
\ \Rightarrow\
n \gtrsim \tfrac{1}{2\varepsilon^2}\ln\tfrac{2}{\delta}.
\]
L’erreur d’EV se borne en multipliant par \(P\) et \(E\).

---

## Remarques

- **Blockers** : on filtre toujours les ranges adverses par \( \mathcal U(h) \).  
- **ICM** : non inclus dans le moteur cEV; conversion possible en post-traitement sur les issues (fold/call/shove).  
- **Extensions** : overcalls (3-way), sizings non all-in, pondérations de combos, etc.

---
