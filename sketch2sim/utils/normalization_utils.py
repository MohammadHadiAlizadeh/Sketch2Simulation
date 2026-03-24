from typing import Any, Dict, List


def build_unit_view_from_structure(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert structure JSON into a unit-centric input/output stream view.
    """
    units: List[Dict[str, Any]] = extraction.get("units", []) or []

    def build_stream(
        stream_id,
        name,
        from_unit,
        to_unit,
        role,
        connect_point=None,
    ):
        return {
            "id": stream_id,
            "name": name or "",
            "from_unit": from_unit,
            "to_unit": to_unit,
            "role": role,
            "connect_point": connect_point,
        }

    all_streams: List[Dict[str, Any]] = []

    for stream in extraction.get("feed_streams", []) or []:
        all_streams.append(
            build_stream(
                stream["id"],
                stream.get("name", ""),
                None,
                stream.get("to_unit"),
                "feed",
                stream.get("connect_point"),
            )
        )

    for stream in extraction.get("intermediate_streams", []) or []:
        all_streams.append(
            build_stream(
                stream["id"],
                stream.get("name", ""),
                stream.get("from_unit"),
                stream.get("to_unit"),
                "intermediate",
                stream.get("connect_point"),
            )
        )

    for stream in extraction.get("product_streams", []) or []:
        all_streams.append(
            build_stream(
                stream["id"],
                stream.get("name", ""),
                stream.get("from_unit"),
                None,
                "product",
                stream.get("connect_point"),
            )
        )

    unit_view_units: List[Dict[str, Any]] = []
    for unit in units:
        unit_id = unit.get("id")
        if not unit_id:
            continue

        streams_in = [
            {
                "id": stream["id"],
                "name": stream.get("name", ""),
                "role": stream.get("role", ""),
                "from_unit": stream.get("from_unit"),
                "connect_point": stream.get("connect_point"),
            }
            for stream in all_streams
            if stream.get("to_unit") == unit_id
        ]

        streams_out = [
            {
                "id": stream["id"],
                "name": stream.get("name", ""),
                "role": stream.get("role", ""),
                "to_unit": stream.get("to_unit"),
                "connect_point": stream.get("connect_point"),
            }
            for stream in all_streams
            if stream.get("from_unit") == unit_id
        ]

        unit_view_units.append(
            {
                "unit_id": unit_id,
                "unit_name": unit.get("name", ""),
                "streams_in": streams_in,
                "streams_out": streams_out,
            }
        )

    return {"units_unit_view": unit_view_units}