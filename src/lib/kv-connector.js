// MultiConnector reads its children in order. PD puts the live transfer
// first and the cache tier second; existing GPU tiers use the default role.
export function multiConnector(connectors, role = "kv_both") {
  return { kv_connector: "MultiConnector", kv_role: role,
    kv_connector_extra_config: { connectors: connectors.filter(Boolean) } };
}
