select 
	cast(cod as varchar) as cod
	,trs.nome_rubrica
	,'' as nome_orgao
	,parcela_unica
	,pode_receber_com_parcela_unica
	,contabiliza_para_teto
	,abate_teto
FROM 
	bdu_sefip.dbo.siape_tipo_rubrica_mapeada strm 
inner join BD_PESSOAL_SIAPE_AMPLO.dbo.tipo_rubrica trs on trs.cod_rubrica = strm.cod
where strm.parcela_unica is not null
and strm.pode_receber_com_parcela_unica is not null
and strm.contabiliza_para_teto is not null
and strm.abate_teto is not null
and (strm.parcela_unica = 1 or strm.pode_receber_com_parcela_unica = 1 or strm.contabiliza_para_teto =1 or strm.abate_teto =1)
and trs.SE_VIGENTE ='S'
UNION
select 
	concat(cast(estrm.cod as varchar), '-',osc.nome)  as cod
	,tres.DESCR_RUBRICA  as nome_rubrica
	,osc.nome as nome_orgao
	,parcela_unica
	,pode_receber_com_parcela_unica
	,contabiliza_para_teto
	,abate_teto
from BDU_SEFIP.dbo.extra_siape_tipo_rubrica_mapeada estrm 
inner join BD_PESSOAL_EXTRA_SIAPE.dbo.TIPO_RUBRICA tres on tres.COD = estrm.cod
left join bdu_sefip.dbo.orgao_extra_siape_clientela osc on osc.cod_clientela = tres.COD_ORGAO 
where parcela_unica is not null
and pode_receber_com_parcela_unica is not null
and contabiliza_para_teto is not null
and abate_teto is not null
and (estrm.parcela_unica = 1 or estrm.pode_receber_com_parcela_unica = 1 or estrm.contabiliza_para_teto =1 or estrm.abate_teto =1)