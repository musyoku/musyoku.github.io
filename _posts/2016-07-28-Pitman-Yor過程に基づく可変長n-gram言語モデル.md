---
layout: post
title: Pitman-Yor過程に基づく可変長n-gram言語モデル（VPYLM）
category: Chainer
tags:
- HPYLM
- VPYLM
excerpt_separator: <!--more-->
---

## 概要

- [Pitman-Yor過程に基づく可変長n-gram言語モデル](http://chasen.org/~daiti-m/paper/nl178vpylm.pdf) を読んだ
- C++でVPYLMを実装した

<!--more-->

## はじめに

VPYLMは[HPYLM](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)を拡張し、HPYLMでは固定だったn-gramオーダーを各単語ごとに推定できるようになったモデルです。

HPYLMと同様に、文脈$h=w_{t-n}...w_{t-1}$に続く単語$w$の確率は

$$
	\begin{align}
		P(w\mid h)=\frac{c(w\mid h) - d\cdot t_{hw}}{\theta + c(h)}+\frac{\theta + d\cdot t_h}{\theta + c(h)}P(w\mid h')
	\end{align}\
$$

となります。

ただし$h'=w_{t-n+1}...w_{t-1}$はn-gramのオーダーを一つ落とした文脈、$c(w\mid h)$はレストラン$h$でのテーブル$w$にいる客の総数、$c(h)=\sum_{w}c(w\mid h)$はその総和、$t_{hw}$は$w$を提供するテーブルの総数、$t_h=\sum_{w}t_{hw}$はその総和を表します。


## 停止確率

VPYLMでは文脈木の各レストラン$u$に、木をルートからたどるときにそこで止まる確率$q_u$があると考えます。

またこれらの停止確率は共通のベータ事前分布から生成されていると仮定します。

$$
	\begin{align}
		q_u \sim {\rm Beta}(\alpha, \beta)
	\end{align}\
$$

また$q_u$の期待値は

$$
	\begin{align}
		\double E[q_u] = \frac{\alpha}{\alpha+\beta}
	\end{align}\
$$

となります。

たとえば単語列$\ The\ quick\ brown\ fox\ jumps\ over\ the\ lazy\ dog\ $があり、$dog$を文脈木に追加する場合を考えます。

深さnのレストランで停止する確率を$q_{n}$とすると、文脈$h$のもとで客$dog$の停止深さが$l$となる確率は

$$
	\begin{align}
		P(n=l\mid h) = q_l\prod_{i=0}^{l-1}(1-q_i)
	\end{align}\
$$

となるので、客$dog$がルートのレストラン$\epsilon$に追加される確率は$q_{\epsilon}$、レストラン$lazy$に追加される確率は$(1-q_{\epsilon})q_{lazy}$、レストラン$the$に追加される確率は$(1-q_{\epsilon})(1-q_{lazy})q_{the}$、・・・のように計算されます。

![VPYLM](/images/post/2016-07-28/vpylm_stop_probs.png)

ちなみに1から引いているのはそこを通過する確率を計算するためです。

通常は深いノードに行くほど停止確率が小さくなりますが、文脈に沿った経路のそれぞれの$q_n$が小さければ深いn-gramとなり、HPYLMとは違って様々な深さのノードを許すモデルになっています。

## 可変長ベイズn-gram言語モデル

VPYLMでは、単語列$\boldsymbol w=w_1w_2...w_T$の確率を

$$
	\begin{align}
		P(\boldsymbol w)=\sum_{\boldsymbol n}\sum_{\boldsymbol \Theta}P(\boldsymbol w, \boldsymbol n, \boldsymbol \Theta)
	\end{align}\
$$

と表します。

$\boldsymbol \Theta$は文脈木の代理客を含めたすべての客の配置を表す隠れ変数、$\boldsymbol n=n_1n_2...n_T$は$\boldsymbol w$のそれぞれの単語が生成された隠れたn-gram長を表します。

HPYLMと同様に客の配置$\boldsymbol \Theta$は推定すべきパラメータであり、VPYLMでは$\boldsymbol \Theta$と$\boldsymbol n$の両方をギブスサンプリングによって推定します。

その際、単語列$\boldsymbol w$の位置$t$の単語$w_t$の隠れたn-gramオーダー$n_t$を、

$$
	\begin{align}
		n_t \sim P(n_t\mid \boldsymbol w, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})
	\end{align}\
$$

のようにギブスサンプリングする必要があるのですが、実際は$P(n_t\mid \boldsymbol w, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})$から直接サンプリングすることはできません。

そこでこの式をベイズの定理から

$$
	\begin{align}
		P(n_t\mid \boldsymbol w, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t}) \propto  
		P(w_t\mid \boldsymbol w_{-t}, \boldsymbol n, \boldsymbol \Theta_{-t})P(n_t\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})
	\end{align}\
$$

と変形します。（$\propto$は比例を意味します）

$\boldsymbol n_{-t}$は、単語$w_t$を除いた単語列$\boldsymbol w_{-t} = w_1,w_2,...,w_{t-1},w_{t+1},...,w_T$に対応するn-gramオーダー$n_1,n_2,...,n_{t-1},n_{t+1},...,n_T$を表しています。

### $n_t$のサンプリング

ここでの$\boldsymbol n_{-t}$はただの整数列であり、この値がそのままモデルのパラメータにあるわけではありません。

そして$n_t$をギブスサンプリングするためには$\boldsymbol n_{-t} = n_1,n_2,...,n_{t-1},n_{t+1},...,n_T$をモデルの何らかのパラメータとして持っている必要があります。

そこでVPYLMでは、文脈木の各レストラン$u$で、全ての単語（客）の通過回数$a_u$と停止回数$b_u$を記録します。

そして式(3)の停止確率$q_u$の期待値をベータ事後分布の期待値として

$$
	\begin{align}
		\double E[q_u]=\frac{a_u+\alpha}{a_u+b_u+\alpha+\beta}
	\end{align}\
$$

と推定します。

こうすることで式(7)の右辺の第二項にある$\boldsymbol n_{-t}$の条件付き確率を

$$
	\begin{align}
		P(n_t=l\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})=\frac{a_l+\alpha}{a_l+b_l+\alpha+\beta}\prod_{i=0}^{l-1}\frac{b_i+\beta}{a_i+b_i+\alpha+\beta}
	\end{align}\
$$

のように表すことができ、$n_t$を$\boldsymbol n_{-t}$の条件付き確率からギブスサンプリングすることができるようになります。

また式(7)の右辺の第一項はn-gramオーダーが$n_t$と決まった時の$w_t$のn-gram確率で、式(1)により計算します。

### $\boldsymbol \Theta$のサンプリング

$\boldsymbol \Theta_{-t}$は単語$w_t$をレストランから除外したあとの客の配置を表します。

これはHPYLMと同様${\rm RemoveCustomer}$をすれば得られます。

その状態で式(7)により新たなn-gramオーダー$n_t$をサンプリングし、その深さ$n_t$に単語$w_t$を追加し直すことで$\boldsymbol \Theta$が自動的にギブスサンプリングされます。

（$\boldsymbol \Theta$は、代理客を含めた客の配置が決定した時点でサンプリングされたことになります。）


## 実装

C++での実装を[GitHub](https://github.com/musyoku/vpylm)にあげておきました。

ハイパーパラメータのサンプリングの実装に関しては[HPYLM](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)の記事に注意点などを書いています。

### $n_t$のサンプリング

$w_t$に対応する$n_t$をサンプリングするには、深さ$l=0,1,2,...$の場合それぞれについて以下の値を求めます。

$$
	\begin{align}
		P(w_t\mid \boldsymbol w_{-t}, \boldsymbol n, \boldsymbol s_{-t})P(n_t=l\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})
	\end{align}\
$$

第一項は式(1)から求めますが、この時深さは$l$と決まっているため、オーダー$l$の文脈$h_l$に対応するレストランを用いて以下のように計算します。

$$
	\begin{align}
		P(w_t\mid h_l)=\frac{c(w_t\mid h_l) - d_{\mid h_l \mid}\cdot t_{h_{l}w_{t}}}{\theta_{\mid h_l \mid} + c(h_l)}+\frac{\theta_{\mid h_l \mid} + d_{\mid h_l \mid}\cdot t_{h_l}}{\theta_{\mid h_l \mid} + c(h_l)}P(w_t\mid h_l')
	\end{align}\
$$

[A Bayesian Interpretation of Interpolated Kneser-Ney](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)の表記に従えば以下のようにも表されます。

$$
	\begin{align}
		P(w_t\mid h_l)=\frac{
			c_{h_{l}w_{t}\cdot} - d_{\mid h_l \mid}\cdot t_{h_{l}w_{t}}
		}{
			\theta_{\mid h_l \mid} + c_{h_{l}\cdot\cdot}
		}+\frac{
			\theta_{\mid h_l \mid} + d_{\mid h_l \mid}\cdot t_{h_l\cdot}
		}{
			\theta_{\mid h_l \mid} + c_{h_l\cdot\cdot}
		}P(w_t\mid \pi(h_l))
	\end{align}\
$$

式(10)の第二項$P(n_t=l\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})$は式(9)から求めますが、この時第二項が十分小さくなれば、それ以降の深さ$l$は見ずに計算を打ち切ります。

また式(9)はレストランが存在しない深さ（つまり通過回数$a_u$や停止回数$b_u$が存在しない）も考慮しなければならないのですが、私はレストランが無い部分では通過回数と停止回数は$0$とし、式(3)を用いて計算しました。

## 実験

今回も[Alice in Wonderland](https://www.cs.cmu.edu/~rgs/alice-table.html)のスペース除去版を用いて文字n-gramの実験を行いました。

### VPYLMとHPYLM

VPYLMを300epochほど学習させたところ、文脈木の一番深いところで$n=11$となったので、比較のため11-gramのHPYLMを学習させてみました。

||VPYLM|HPYLM|
|n|-|11|
|パープレキシティ|8.84407|3.58401|
|ノード数|27075|34841|
|総客数|218630|439133|

また文章生成を行ってみました。

VPYLM

```
^Don'tlethingsaid,`Evidence,'saidtheCaterpillar.$
^`Holdyoubythefirstsentencefeltthatitwasinthatsherandthen,'thoughtAlice,`ifyouwouldn'tbehurriedout,aller,them.$
^Heretheyhumblingveryangtoseeifshewasreadytosingthem!$
^However,shesoonmakeone,andshewenton,`--you,andevenifmilingingherself,`andthen,andmakeoutwhointhedistance,'saidtheCaterpillar.$
^Alicewhiskers!'$
^TheDuchess,whoonlythroat!'ontothebottomofawell?'saidtheCaterpillar.`Isthathehallhim.$
^`That'sthematterwentslowlybe,'itis.$
^Afterawastrangether.$
^You'reewithyouwouldn'tbeguntoday,`forI'llhimTobefoundherarm,andallthewithonefingerto-day?'$
^Asifyouwouldn'tbetternow,whichshehadbeenthemomenttreesuchacurioustodream,'saidtheCat,'saidtheCaterpillar.$
^AlicethoughtAlice,`Oh,ItellyoujustthenshewassoonasitcriedtheMockTurtlewouldgetreesunderstanddownintoaskedintotheGryphon,`--youeverseemtogoneachshehadbeenwandthen,`andthen,whenshrilllookedsomealittlefthandbitoffall,lyingdownhissonstailaboutagaininarycurioustodream,sethat!'onthatsherememberherarm,andallthewood.$
^`NotI'maporpoisonlyapacewasveryhands,andbeganthat,'saidtheCaterpillar.$
^`Yes,it'sanatomakeoneofthembowedhisveryhadtoaskhishead!$
^Presentstandgladtofindthat'sthemomenttrouble!'$
^TheDuchess,'saidtheCatreesomewhere.$
^Heretheycouldn'tlikethelookedsomealittleway,'added,andwasjustintime,'thoughtAlice,`Doyouthinkingreatcrossedbysalittlesisters,orthunderthansweresay?'$
^`Itwasoutofthewhileftoff.'Afterallyabouting`Well,thatsheran;andbegantage,asshecouldn'tlikethelookedsomewhere.$
^Comeon!'$
^Oh,myplandontheirface.$
^`NotI'mnotanevertomakeoneofthembowagain,forthis,soshetuckedherself,`tillthewholeuseofthembowagain,forthis,soshetuckedherself,`tilltheway,'andsnearly,`It'sthem!$
^`Iwonderingwildlikethelookedsomebodytocats.'$
^`It'sthem,andafterthanswerenoafewminute,there'snothingsaid,`Everybodylivethecourtwasobear:itpoorAlice,`ifyis,youknow.'$
^Buthere?$
^`Itisn'tarts,$
^"Themouthere?$
```

HPYLM(11-gram)

```
^`Whichseemedtoneofgreatureshallhaveputthetookmeforpoise,"Keepy;`andtheMouseofalledtheRabbingandverybodysay,'saidtheDormouse,andIshallforshecondthedoor,halfoffright-eyedterrilyandsheverhapsyoufly,Ishallspeech,there?$
^ShewalkingtoAlicesallthreegarden."'$
^OhdearpawsinsuchacurioustodyandshesaidtheyoutofallingmesseduptoAlice:`allIbegyoulike,`was,thatshewasverycurtanttonothegame,cape!'sheasky-rockettle,`thee--howis,shesatoneonlyknowthem,andshefeltquitefoldenkey,thoughtAlicedoutaporpoise--'$
^"Upabovehisheadpresenton,`andtheaccidedlywenthedistancetohersenouseagain,inasortofknot,wouldhavenoticetherownonherhapsitwasattheMockTurtle,tilljustnow?'$
^(Weknow.'$
^`Onlymakesyouwereornameliketobeginninefeeblast,andmakindownhairthatinasolemntonguehand.$
^`No,I'veoff,andthinkyou'remarkable,andAliceafterwhenIgrowhere--'$
^TheQueens,andwentAlice:`Iheardhim.$
^TellheratsavaguessedwhoisDinahurriedintoabutteringheardavoice,verygladtosay,`Dobats?$
^Butheright,hurrytoldyoubutthetimewhenthewantyofspeaktothehadplendindeed:shefeltsurprisedtoexecutiouscreatealargecandrawwaterpillar.$
^AlicewasoveralittleBill,whichwasmovingrilyabout,andlooking,andstupidly.$
^`Oh,youknow?$
^Itdidnothinkinganxiouslytookupandthebook,'saidthetimebusyfarm-in-the-bye,whenhissedtohersnow?'$
^Don'tyourMajesty,'shead!$
^Atlastsaytoher,aboutsomesatsifyouwon'tlikedtheCat$
^`IfanythingIeverysleep$
^`Nothinkhowgladshehadputbackagain,forsomethesameofthem,andtheDuchess;`andnonsense.$
^I'llneverywhiteRabbit-holewithcupbothcreatabat?'inquiredherwithitsmouthssoverywellsay,'saidAlice,brokengladI'vesee:I'lltryingtogetitwouldbenochangingtoitseyesanxioustokneeatlysaidthemraw.'$
^`Iwishyouwereplied.$
^Theadsareputtingingclubs;theDucheshireCaterpillar.$
^`Iwon't,'saidthebirdasshecoolfound.$
^`Andyetitwaterpillar,justatpresense.'$
^Oh,Ibegyourwaited.$
```

次に、ある文字がどの深さから生成されたかを表示します。

たとえば

```
Alice
01234
```

ならeはAlicから生成されたということを表しています。

```
^TheQueen!'andthethreegardenersinstantlythrewthemselvesflatupontheirfaces.$
012231344432336454554345445435543433665322333233432352333323343344643334301

^`That'sthereasonthey'recalledlessons,'theGryphonremarked:`becausetheylessenfromdaytoday.'$
0122334333445465434433222445453333544412454233434423453333122434335455323433442352335433312

^Improvehisshiningtail,$
012233424333343444633332

^`Itisalongtail,certainly,'saidAlice,lookingdownwithwonderattheMouse'stail;`butwhydoyoucallitsad?$
01223353343333431234445432124435233361234554323532334333443443452244323044312233324423333433044341

^Down,down,down.$
01234444443233422

^TheQueenturnedangrilyawayfromhim,andsaidtotheKnave`Turnthemover!'$
0123415343423453234454242322234233323444356434412334223355533333522

^Imaginehersurprise,whentheWhiteRabbitreadout,atthetopofhisshrilllittlevoice,thename`Alice!'$
011234343344433443532223354433445124224345334432344754534333335546343443324343534344522333722

^Allpersonsmorethanamilehightoleavethecourt.'$
0123323445323433436444335443443344434455465422

^ThepoorlittleLizard,Bill,wasinthemiddle,beingheldupbytwoguinea-pigs,whoweregivingitsomethingoutofabottle.$
01223233643443322333403244244354446343433343351334444222303334463233312322444322334332334455534345353334331

^`Nothingwhatever,'saidAlice.$
012233455424334254124535243361

^But,whenthetiderisesandsharksarearound,$
01223022544445334445354455343323542544661

^`Come,weshallhavesomefunnow!'thoughtAlice.`I'mgladthey'vebegunaskingriddles.--IbelieveIcanguessthat,'sheaddedaloud.$
012223432434434234433333393343233463462233612233234434632235334335233333333362232343532012334344454441234445343243442

^`Areyoutogetinatall?'saidtheFootman.`That'sthefirstquestion,youknow.'$
01222442563434544334322343563323343341223342544444444134475333334234522

^Andthemuscularstrength,whichitgavetomyjaw,$
01233456333243344422334312454353233363332233

^`Youare,'saidtheKing.$
01222434422343534512351

^Therearenomiceintheair,I'mafraid,butyoumightcatchabat,andthat'sverylikeamouse,youknow.$
0123344544433435344456343122433344134234533343233434433223334453023542443333454123423651

^Andhavegrownmostuncommonlyfat;$
01233334443335433333345434322241

^`Well,Ishan'tgo,atanyrate,'saidAlice:`besides,that'snotaregularrule:youinventeditjustnow.'$
01222442223363323433344344512443522334222544353335452333434434334523312353333343345233333042

^`That'senoughaboutlessons,'theGryphoninterruptedinaverydecidedtone:`tellhersomethingaboutthegamesnow.'$
01223342444444144734344443412334423433334454433533644134534333434544122344234533435453333344433533424323

^`Areyoucontentnow?'saidtheCaterpillar.$
0122234333454233344134455636243454335462

^Therewasageneralclappingofhandsatthis:itwasthefirstreallycleverthingtheKinghadsaidthatday.$
01233444555433465424034341433353235435412334344454434335442336544435535431233333233543435423

^AtlasttheGryphonsaidtotheMockTurtle,`Driveon,oldfellow!$
012344454442334334433564345234422343312324243422343345431

^`Comeon!'criedtheGryphon,and,takingAlicebythehand,ithurriedoff,withoutwaitingfortheendofthesong.$
01223343452333444352344335133313353433243332344534322344343454431233435324334532345343444544433441

^`Yourhairwantscutting,'saidtheHatter.$
012323434333334324443432235356343334571

^Alicewenttimidlyuptothedoor,andknocked.$
01234362434323334243333456434343022440332

^That'llbeacomfort,oneway--nevertobeanoldwoman--butthen--alwaystohavelessonstolearn!$
0122342232333433343224424443232453433353243344325435445322232443354434324545454344434

^ThePanthertookpie-crust,andgravy,andmeat,$
0123314344644354233253235423333452223323531

^FatherWilliamstandingonhead$
01234450333567333454444442344

^`Butwhathappenswhenyoucometothebeginningagain?$
012235523354446454234334334534345433463443333441

^`Somebodysaid,'Alicewhispered,`thatit'sdonebyeverybodymindingtheirownbusiness!'$
012123454334236122333645344444532234463232344325345435332244343344434452436545422

^`Readthem,'saidtheKing.$
0123444343424444564522332

^Comeon!'$
0123343412

^Quick,now!$
011333322543

^`Oh,there'snouseintalkingtohim,'saidAlicedesperately:`he'sperfectlyidiotic!$
01232124446343444533235444323334323435233343735444547232233524434343323433632

^Dodopresentingthimble$
01224344444543555324333

^Aliceconsideredalittle,andthensaid`Thefourth.'$
012333433354344754444433333444553335223345345322

^`Aswetasever,'saidAliceinamelancholytone:`itdoesn'tseemtodrymeatall.'$
01212443234244123535223333353634455342335412333425433334423334334454512

^`Don'tbeimpertinent,'saidtheKing,`anddon'tlookatmelikethat!$
0121332343344534344342244366551233522333443233433333454444551

^Alicerepliedeagerly,forshewasalwaysreadytotalkaboutherpet:`Dinah'sourcat.$
012233445344443444243163455454343444333454343442333444434331232343233346433

```

## 関連

- [A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)
- ベイズ階層言語モデルによる教師なし形態素解析（準備中）
- 隠れセミマルコフモデルに基づく教師なし完全形態素解析（準備中）
