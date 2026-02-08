from dataclasses import dataclass

import pytest
import regex

from llm_py.tokenizer import Chunk, Pair, get_merges_inplace

TEST_CORPUS = """
Shipment & Transport-Sea, Air, Rail, Road, Pipeline
The mode of transportation is an important consideration when planning the shipment process. Besides the costs, the urgency of the shipment, the value of the goods being shipped as well as the size and weight of the goods need to be evaluated when determining the form of transportation.
Seaborne trade accounts for about 90% of the global trade, and as per UNCTAD, 1687 million tons (2015 estimate) were carried in around 177.6 million containers (2015 estimate) covering 998 billion ton-miles (2016 estimate).
Because of size or volume, there are several types of cargoes that cannot be or is economically unviable to move by other modes of transport than the sea.
Ocean freight is a less expensive method of shipping goods, but the drawback is a longer transit time. Another benefit for ocean freight is while size and weight may be an issue for air; it is not for ocean freight.
Ocean freight is used quite extensively for the movement of bulk commodities such as agri-products (wheat, maize, soya, etc.), coal, iron ore or for wet bulk products such as crude oil and petroleum. Also, larger, odd-shaped items including engines and propellers may move via this mode as well, depending on how sensitive the delivery time is.
Ocean freight is also a preferred mode of transport for the movement of high volume and heavy cargo such as minerals, metals, ores, steel coils, etc. which would be impossible to move by air freight.
Additionally, businesses are placing more of an emphasis on the environmental impact on shipping. An air freight service emits a higher amount of polluting gases with less space capacity compared to sea freight services which are considered a much greener transportation mode with a higher carrying capacity.
Key benefits of ocean freight include
- Suitable for wide range of products with long lead times
- Large volumes. A single, ultra-large container ship can carry +/-20,000 twenty-foot equivalent units (TEU)
- Most environmental friendly among all modes of transport
- Liner shipping is the most efficient mode of transport for goods
- Extensive coverage around the world
- Multiple carrier options for the shippers
Over the next 15 years, as the world GDP grows, there will be a demand for higher value goods. As per Boeing’s 2016 – 2017 world air cargo forecast, there will be a proportionate growth in the value per ton of total traded goods around the world.
To meet the demand for growth, world air cargo traffic is forecasted to grow an average 4.2 percent per year.
Air freight is a critical mode of transport. It serves markets and supply chains that demand speed. One of greatest examples goes back to 1997 when Apple began innovating on the nitty-gritty details of supply-chain management. Almost immediately upon Steve Jobs’ return. At the time, most computer manufacturers transported products by sea, a far cheaper option than air freight.
Steve Jobs took advantage of the benefit of air freight and used an innovative strategy. He paid $50 million to buy up all the available holiday air freight space to ensure that the company’s new, translucent blue iMacs would be widely available during Christmas season giving them a massive competitive advantage over their rivals. – “It was an ‘Oh s—’ moment,” recalls former HP supply chain chief Mike Fawkes.”
Other industries such as the automotive and retail industry also utilize air freight to achieve ‘just-in-time’ (JIT) inventory replenishment. JIT option allows stores, production lines to place order fulfillment based on demand as, and when required. It provides greater flexibility and reduces inventory and storage costs.
Also, perishable goods such as foods, flowers, and some pharmaceuticals also take advantage of shorter transit time. Another positive for air freight is that there’s less handling of cargo overall, so the likelihood of damage or theft is less likely when utilizing air.
But air freight also has its own disadvantages such as being one of the most expensive due to the requirement of speed and the fuel that is used.
It also has its size and weight limitations. Regulatory bodies limit what can and cannot be transported by air, and as such, oddly shaped or very large items may be more suitable for other modes of transport.
Key benefits of air freight include
- Quick transit
- Less handling of cargo
- Less documentation
- Reliable arrival and departures
- Enhanced level of security for your cargo
Another mode of transport which is also considered a ‘green’ option is rail. Trains burn less fuel per ton-mile than road vehicles and a train, which can have as many than 100 wagons, only needs one driver. There are, however, some additional costs which are incurred in a rail journey: at each end of the rail transit, a road delivery will be needed, and there will be a lift cost to transfer the container between the train and the road vehicle.
On average, longer journeys tend to be less expensive by rail, and shorter journeys are less costly by road. Where the point of cost neutrality comes is governed by many factors which are route and commodity specific, but in general, the point of cost neutrality can be expected to lie in the range of 130 to 150 miles.
In 2015, the first freight train carrying ISO freight containers from China arrived in the Port of Rotterdam in 18 days as against the normal 44 odd days by the sea.
This movement of containerized cargo by rail from China to logistics hubs in Europe such as in the Netherlands, UK is seen as a significant step in the development of trade between the two continents. It has encouraged multinationals such as Hewlett-Packard and Ricoh to use the route from Europe to China for their cargoes.
The Manager of European Transport at Ricoh notes that if one can set up an effective planning, rail is a relatively quick mode of transport taking only 20 days to China. In addition, the move by rail also has some advantages such as all containers being transported to the location in one go, while being environmentally friendly as a train releases far less CO2 than a plane.
Key benefits of rail freight include
- Reliable transit times and schedules
- Railroads are the most efficient form of land transportation. One train can haul the equivalent of over 400 trucks
- Fast and cost-effective deliveries over long distances. Typically over 500 miles
- Traditionally, rail has a strong safety record.
- Helps in alleviating road congestion, thus lowering emissions
Road freight is one of the most common of all modes of transportation. It is widely used in continents such as Europe, Africa, and North America. The single customs document process provides a seamless movement of goods even across various states and countries.
Road freight provides several advantages over other modes of transportation such as
- Quick and scheduled delivery
- Local, over border, long or short haul deliveries even in rural areas
- Flexible service
- Saving in Packing Cost compared to other modes
- Track and trace of cargo and truck
- Complete door-to-door service and it is one of the more economical means of transport.
However, truck transport is limited somewhat as to what it can carry by the size of the vehicles used and by size and weight restrictions. Another limitation is that it is affected by weather, road conditions and traffic.
Pipeline transport is the long-distance transportation of a liquid or gas through a system of pipes—a pipeline—typically to a market area for consumption. The latest data from 2014 gives a total of slightly less than 2,175,000 miles (3,500,000 km) of pipeline in 120 countries of the world. The United States had 65%, Russia had 8%, and Canada had 3%, thus 75% of all pipeline were in these three countries.
Pipeline and Gas Journal’s worldwide survey figures indicate that 118,623 miles (190,905 km) of pipelines are planned and under construction. Of these, 88,976 miles (143,193 km) represent projects in the planning and design phase; 29,647 miles (47,712 km) reflect pipelines in various stages of construction. Liquids and gases are transported in pipelines and any chemically stable substance can be sent through a pipeline. Pipelines exist for the transport of crude and refined petroleum, fuels – such as oil, natural gas and biofuels – and other fluids including sewage, slurry, water, beer, hot water or steam for shorter distances. Pipelines are useful for transporting water for drinking or irrigation over long distances when it needs to move over hills, or where canals or channels are poor choices due to considerations of evaporation, pollution, or environmental impact.
""".strip()

GPT4_SPLIT_PATTERN = r"""
    # shorten form of words like "will" ('ll), "have" ('ve), etc
    '(?i:[sdmt]|ll|ve|re)
    # 
    |[^\r\n\p{L}\p{N}]?+\p{L}+
    # numbers between 0 - 999
    |\p{N}{1,3}
    |[ ]?[^\s\p{L}\p{N}]++[\r\n]*
    |\s*[\r\n]
    |\s+(?!\S)
    |\s+
    """


@dataclass
class ChunkNaive:
    bytes: list[int]
    weight: int


def get_merges_naive(chunks: list[ChunkNaive], n_merges: int):
    merges: list[Pair] = []
    for i in range(n_merges):
        pair_counts: dict[Pair, int] = {}
        for chunk in chunks:
            for pair in zip(chunk.bytes, chunk.bytes[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + chunk.weight

        max_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
        # print(f"{i} SLOW pair_counts {max_pair}")
        # for pair, count in pair_counts.items():
        #     print(pair, "->", count)

        new_pair_idx = 256 + i
        merges.append(max_pair)

        for chunk in chunks:
            new_tokens: list[int] = []
            j = 0
            while j < len(chunk.bytes):
                if (
                    j < len(chunk.bytes) - 1
                    and chunk.bytes[j] == max_pair[0]
                    and chunk.bytes[j + 1] == max_pair[1]
                ):
                    new_tokens.append(new_pair_idx)
                    j += 2
                else:
                    new_tokens.append(chunk.bytes[j])
                    j += 1
            chunk.bytes = new_tokens

    return merges


@pytest.mark.parametrize("n_merges", [10, 100, 100])
def test_tokenizer_train(n_merges: int):
    split_pattern = regex.compile(GPT4_SPLIT_PATTERN, regex.VERBOSE)

    chunk_count: dict[str, int] = {}
    for chunk in split_pattern.findall(TEST_CORPUS):
        chunk_count[chunk] = 1 + chunk_count.get(chunk, 0)

    expected_chunks = [
        ChunkNaive(list(chunk_bytes.encode("utf-8")), weight)
        for chunk_bytes, weight in chunk_count.items()
    ]
    expected_merges = get_merges_naive(expected_chunks, n_merges)
    expected_pair_count: dict[Pair, int] = {}
    for chunk_idx, chunk in enumerate(expected_chunks):
        for exp_pair in zip(chunk.bytes, chunk.bytes[1:]):
            expected_pair_count[exp_pair] = (
                expected_pair_count.get(exp_pair, 0) + chunk.weight
            )

    actual_chunks: list[Chunk] = [
        (list(chunk_bytes.encode("utf-8")), weight)
        for chunk_bytes, weight in chunk_count.items()
    ]
    actual_pair_count: dict[Pair, int] = {}
    pair_loc: dict[Pair, set[int]] = {}
    for i, (chunk_bytes, chunk_weight) in enumerate(actual_chunks):
        for exp_pair in zip(chunk_bytes, chunk_bytes[1:]):
            if exp_pair not in pair_loc:
                pair_loc[exp_pair] = set()
            actual_pair_count[exp_pair] = (
                actual_pair_count.get(exp_pair, 0) + chunk_weight
            )
            pair_loc[exp_pair].add(i)

    actual_merges = get_merges_inplace(
        actual_chunks, actual_pair_count, pair_loc, n_merges
    )

    if all(
        exp[0] == act[0] and exp[1] == act[1]
        for exp, act in zip(expected_merges, actual_merges)
    ):
        return

    # for fast_chunk, slow_chunk in zip(fast_chunk_list, slow_chunk_list):
    #     fast_chunk_bytes, _ = fast_chunk
    #     if len(fast_chunk_bytes) != len(slow_chunk.bytes) or any(
    #         fast_byte != slow_byte
    #         for fast_byte, slow_byte in zip(fast_chunk_bytes, slow_chunk.bytes)
    #     ):
    #         print("chunk mismatch")
    #         print(fast_chunk_bytes)
    #         print(slow_chunk.bytes)

    for exp_pair, exp_count in expected_pair_count.items():
        assert exp_pair in actual_pair_count, (
            f"{exp_pair} | Missing pair with count {exp_count}"
        )

        act_count = actual_pair_count[exp_pair]
        assert exp_count == act_count, (
            f"{exp_pair} | Expecting count {exp_count} but gotten {act_count}"
        )

    assert len(expected_pair_count) == len(actual_pair_count), (
        f"num pair mismatch, expecting {len(expected_pair_count)} but gotten {len(actual_pair_count)}"
    )
