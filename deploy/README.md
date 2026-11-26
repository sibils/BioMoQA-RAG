# Deployment

**The live Ansible does not live here.** It lives in
[bitem-heg-geneve/ansible-cfg](https://github.com/bitem-heg-geneve/ansible-cfg):

| Environment | Host | Playbook in ansible-cfg | Branch |
|---|---|---|---|
| dev | `sibils-dev-ai.lan.text-analytics.ch` | `playbook_dev_sibils_qa.yml` | `master` |
| prod | `sibils-prod-ai.lan.text-analytics.ch` | `playbook_prod_sibils_qa.yml` | `prod_5.1.5.10` |

Each playbook ships its `config.toml` through the role's `static_file_dir`, from
`files/<playbook name>/config.toml` in that repo. Jenkins runs the playbooks.

The models are pinned artifacts there (`bitem.artifacts`, ansible-cfg PR 143); the service
downloads nothing. The mirror of the prod files that this directory used to hold is gone:
read them in ansible-cfg.

This directory previously held a second, hand-rolled Ansible setup (uv, `/opt/biomoqa`)
that was never used and had drifted from reality. It has been removed rather than left to
mislead the next person.

See [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) for the promotion flow and the config traps.
